import random


class Posl(object):

    def kk(self, ot, do, max):
        random.seed()
        k = random.randint(ot, do)
        ll = [i for i in range(max)]
        random.shuffle(ll)
        return ll[:k]
