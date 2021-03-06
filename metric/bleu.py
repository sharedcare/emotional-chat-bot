from .bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"

if __name__ == "__main__":
    score = Bleu(4)
    # read the file
    model = 'DSHRED'
    dataset = 'ubuntu'
    path = f'../processed/{dataset}/{model}/pred.txt'

    with open(path) as f:
        refs , tgts = [], []
        for idx, line in enumerate(f.readlines()):
            if idx % 4 == 0:
                pass
            elif idx % 4 == 1:
                refs.append(line.strip()[13:])
            elif idx % 4 == 2:
                tgts.append(line.strip()[13:])
            else:
                pass
    refs = {idx: [line] for idx, line in enumerate(refs)}
    tgts = {idx: [line] for idx, line in enumerate(tgts)}
    ss = score.compute_score(refs, tgts)
    print(ss[0])