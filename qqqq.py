import random


class Posl(object):

    # def bas(self, x):
    #     if x == 3:
    #         i, j, k = 0, 1, 2
    #     elif x == 2:
    #         i, j, k = 0, 0, 1
    #     else:
    #         i, j, k = 0, 0, 0
    #     qq = []
    #     while i < 6:
    #         while j < 7:
    #             while k < 8:
    #                 if x == 3:
    #                     qq.append(bin(2 ** i + 2 ** j + 2 ** k)[2:])
    #                 elif x == 2:
    #                     qq.append(bin(2 ** k + 2 ** j)[2:])
    #                     i = 10
    #                 else:
    #                     qq.append(bin(2 ** k)[2:])
    #                     j = 10
    #                     i = 10
    #                 while len(qq[len(qq) - 1]) < 8:
    #                     qq[len(qq) - 1] = "0" + qq[len(qq) - 1]
    #                 k += 1
    #             j += 1
    #             k = j + 1
    #         i += 1
    #         j = i + 1
    #         k = j + 1
    #     return qq
    #
    # def dec(self, s):
    #     ind = []
    #     for i in range(len(s)):
    #         if s[i] == '1':
    #             ind.append(i)
    #     return ind

    def kk(self, ot, do, max):
        k = random.randint(ot, do)
        l = [[random.randint(0, max-1), i] for i in range(max)]
        l.sort()
        return [i[1] for i in l]
