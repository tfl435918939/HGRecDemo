#!/usr/bin/python
# coding:utf-8
import numpy as np
import time
import scipy.sparse as ss


class metapathGeneration:
    # u->e b->p ca->d ci->dt
    def __init__(self, enum, pnum, dnum, dtnum):
        self.unum = enum + 1
        self.bnum = pnum + 1
        self.canum = dnum + 1
        self.cinum = dtnum + 1
        ep = self.load_ub('../data/ep_0.8.train')
        # 获得EPE
        # self.get_UBU(ep, '../data/metapath/epe_0.8.txt')
        # 获得EPDPE
        # self.get_UBCaBU(ep, '../data/pd.txt', '../data/metapath/epdpe_0.8.txt')
        # 获得EPDtPE
        self.get_UBCiBU(ep, '../data/pdt.txt', '../data/metapath/epdtpe_0.8.txt')
        # 获得PEP
        self.get_BUB(ep, '../data/metapath/pep_0.8.txt')
        # 获得PDtP
        self.get_BCiB('../data/pdt.txt', '../data/metapath/pdtp_0.8.txt')
        # 获得PDP
        self.get_BCaB('../data/pd.txt', '../data/metapath/pdp_0.8.txt')

    # 1.将u-b矩阵中的值初始化为0
    # 2.从文件中读取u-b的评分数据，矩阵对应位置置为1
    def load_ub(self, ubfile):
        ub = np.zeros((self.unum, self.bnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                ub[int(user)][int(item)] = 1
        ub = ss.csc_matrix(ub)
        return ub

    def get_UBU(self, ub, targetfile):
        print('EPE adjacency matrix multiplication ...')
        uu = ub.dot(ub.T)
        uu = self.sparse2dense(uu, u'epe_0.8_sparse')
        print(uu.shape)

        print('writing to file...')
        self.save(targetfile, uu)

    def get_BUB(self, ub, targetfile):
        print('PEP adjacency matrix multiplication...')
        mm = ub.T.dot(ub)
        mm = self.sparse2dense(mm, u'pep_0.8_sparse')
        print(mm.shape)

        print('writing to file...')
        self.save(targetfile, mm)

    def get_BCiB(self, bcifile, targetfile):
        print('PDtP adjacency matrix initialization..')
        bci = self.matrix_init(bcifile)

        print('PDtP adjacency matrix multiplication..')
        mm = bci.dot(bci.T)
        mm = self.sparse2dense(mm, u'pdtp_0.8_sparse')

        print('writing to file...')
        self.save(targetfile, mm)

    def get_BCaB(self, bcafile, targetfile):
        print('PDP adjacency matrix initialization..')
        bca = self.matrix_init(bcafile)

        print('PDP adjacency matrix multiplication..')
        mm = bca.dot(bca.T)
        mm = self.sparse2dense(mm, u'pdp_0.8_sparse')

        print('writing to file...')
        self.save(targetfile, mm)

    def get_UBCaBU(self, ub, bcafile, targetfile):
        print('EPDPE adjacency matrix initialization..')
        bca = self.matrix_init(bcafile)

        print('EPDPE adjacency matrix multiplication...')
        uu = ub.dot(bca).dot(bca.T).dot(ub.T)
        uu = self.sparse2dense(uu, 'epdpe_0.8_sparse')

        print('writing to file...')
        self.save(targetfile, uu)

    def get_UBCiBU(self, ub, bcifile, targetfile):
        print('EPDtPE adjacency matrix initialization..')
        bci = self.matrix_init(bcifile)

        print('EPDtPE adjacency matrix multiplication...')
        uu = ub.dot(bci).dot(bci.T).dot(ub.T)
        uu = self.sparse2dense(uu, 'epdtpe_0.8_sparse')

        print('writing to file...')
        self.save(targetfile, uu)

    def sparse2dense(self, matrix, filename):
        np.save('../data/metapath/' + filename, matrix)
        matrix = np.load('../data/metapath/' + filename + '.npy')[()]
        matrix = matrix.toarray()
        return matrix

    def matrix_init(self, file):
        matrix = np.zeros((self.bnum, self.cinum))
        with open(file, 'r') as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                matrix[int(m)][int(d)] = 1
        sparse_matrix = ss.csc_matrix(matrix)
        return sparse_matrix

    def save(self, targetfile, matrix):
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(matrix.shape[0])[1:]:
                for j in range(matrix.shape[1])[1:]:
                    if matrix[i][j] != 0 and i != j:
                        print('line->' + str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n ')
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(matrix[i][j])) + '\n')
                        total += 1
        print('total = ', total)


if __name__ == '__main__':
    # see __init__()
    metapathGeneration(enum=31868, pnum=24225, dnum=3541, dtnum=28)
