package kwdx

import (
	"github.com/alixaxel/pagerank"
	"gonum.org/v1/gonum/mat"
	"gopkg.in/jdkato/prose.v2"
	"sort"
	"strings"
)

// Sieve extracts keywords from a given prose Document.
type Sieve struct {
	Stopwords map[string]struct{}
	Embed     func(string) mat.Vector
}

// SetStops sets the stopwords of the Sieve. For more advanced set union,
// difference, and other operations, please initialize this value directly.
func (xtr *Sieve) SetStops(stops ...string) {
	xtr.Stopwords = make(map[string]struct{}, len(stops))
	for _, t := range stops {
		xtr.Stopwords[strings.ToLower(t)] = struct{}{}
	}
}

// New instantiates a new Sieve.
func New(embed func(string) mat.Vector, stops []string) *Sieve {
	xtr := new(Sieve)
	xtr.Embed = embed
	if stops == nil {
		stops = []string{}
	}
	xtr.SetStops(stops...)
	return xtr
}

// Keywords stores the end product of the keywordization process in sorted
// order. Implements `sort.Interface`.
type Keywords struct {
	Tokens   []string
	Rankings []float64
}

// RankMap returns an associative mapping of terms to their centrality.
func (K Keywords) RankMap() map[string]float64 {
	l := K.Len()
	rtn := make(map[string]float64, l)
	for i := 0; i < l; i++ {
		rtn[K.Tokens[i]] = K.Rankings[i]
	}
	return rtn
}

func (K Keywords) Len() int {
	k := len(K.Tokens)
	r := len(K.Rankings)
	if k < r {
		r = k
	}
	return r
}

func (K Keywords) Less(i, j int) bool {
	return K.Rankings[i] < K.Rankings[j]
}

func (K Keywords) Swap(i, j int) {
	tmpf := K.Rankings[i]
	K.Rankings[i] = K.Rankings[j]
	K.Rankings[j] = tmpf

	tmps := K.Tokens[i]
	K.Tokens[i] = K.Tokens[j]
	K.Tokens[j] = tmps
}

// Sift uses the cosine similarities of unique terms' embeddings in the
// document to rank the centrality of terms, exploiting frequency of
// topically-relevant siblings in a bag-of-words to learn separate central
// ideas or entities in a document through PageRank and word embeddings.
// Ignores those terms for which no word vector can be found (xtr.Embed returns
// `nil`).
func (xtr Sieve) Sift(document string) (K Keywords) {
	D, _ := prose.NewDocument(document)
	K.Tokens = make([]string, 0, len(D.Tokens()))
	K.Rankings = make([]float64, len(D.Tokens()))
	for _, t := range D.Tokens() {
		k := strings.ToLower(t.Text)
		if _, ok := xtr.Stopwords[k]; ok {
			continue
		}
		K.Tokens = append(K.Tokens, k)
	}
	G := pagerank.NewGraph()
	for i, k := range K.Tokens {
		for j, t := range K.Tokens {
			if t == k {
				continue
			}
			a := xtr.Embed(t)
			if a == nil {
				continue
			}
			b := xtr.Embed(k)
			if b == nil {
				continue
			}
			weight := mat.Dot(a, b) / mat.Norm(a, 2) / mat.Norm(b, 2)
			G.Link(uint32(i), uint32(j), weight)
		}
	}
	G.Rank(0.85, 1e-6, func(i uint32, rank float64) {
		K.Rankings[int(i)] = rank
	})
	K.Rankings = K.Rankings[:len(K.Tokens)]
	sort.Sort(K)
	return K
}
