# -*- coding: utf-8 -*-
import numpy as np
import random


class DEAlgorithm:
    def __init__(self,config):
        self.config = config

    def initialization(self,initial_popusize):
        '''
        initialize population
        :return: initialized population
        '''
        ArryLi = []
        for i in range(initial_popusize):
            initLi = []
            for bound in self.config.bounds_all:
                init = np.random.uniform(low=bound[0], high=bound[1], size=1)
                initLi.append(init[0])
            ArryLi.append(initLi)

        return ArryLi

    def listminus(self,l1,l2):
        '''
        two list minus
        :param l1: list 1
        :param l2: list 2
        :return: minus list
        '''
        out = [a-b for a,b in zip(l1,l2)]
        return out

    def listadd(self,l1,l2):
        '''
        function to add two list
        :param l1: list1
        :param l2: list2
        :return: added list
        '''
        out = [a+b for a,b in zip(l1,l2)]
        return out

    def multply(self,a,li):
        '''
        function to multply two lists a*list
        :param a: a scalar
        :param li: a list
        :return: a list
        '''
        out = [a*b for b in li]
        return out

    def checkin(self,value,bound):
        if value>=bound[0] and value<=bound[1]:
            return True
        else:
            return False

    def inrange(self,list):
        '''
        determine the initialized population is in the bounds
        :param mutated: population after mutation
        :return: True or False
        '''
        logicli = []
        for i in range(len(list)):
            value = list[i]
            bound = self.config.bounds_all[i]
            logic = self.checkin(value,bound)
            logicli.append(logic)
        logic_ = all(logicli)

        return logic_

    def mutation(self,InitialArray,F_l,F_u):
        '''
        mutation function
        :param InitialArray: initialized population
        :return: mutated population
        '''
        ArryLi = InitialArray.copy()
        for i in range(len(ArryLi)):
            Contin = True
            while Contin:
                F = np.random.uniform(low=F_l, high=F_u, size=1)[0]
                idxlist = list(range(len(ArryLi)))
                idxlist.remove(i)
                idxLI = random.sample(idxlist, 3)
                mu_a = self.listminus(ArryLi[idxLI[1]],ArryLi[idxLI[2]])
                mu_b = self.multply(F,mu_a)
                mu_c = self.listadd(ArryLi[idxLI[0]],mu_b)
                if self.inrange(mu_c):
                    ArryLi[i] = mu_c
                    Contin = False

        return ArryLi

    def crossover(self,InitialArray,MutatedArray):
        '''
        crossover function
        :param InitialArray: initialized population
        :param MutatedArray: mutated population
        :return: crossovered population
        '''
        CrossArray = []
        for i in range(len(InitialArray)):
            arry = []
            for j in range(len(InitialArray[i])):
                Fc = np.random.uniform(low=0, high=1, size=1)[0]
                if Fc >= self.config.F_c:
                    arry.append(MutatedArray[i][j])
                else:
                    arry.append(InitialArray[i][j])

            CrossArray.append(arry)

        return CrossArray

    def selection(self, gmeanlist, InitialArray, CrossArray, k):
        '''
        selection function
        :param gmeanlist: fitness list
        :param InitialArray: initialized population
        :param CrossArray: crossovered population
        :return: selectArray: selected population; Bestgmean: best fitness value (gmean)
        '''
        Bestgmean = []
        selectArray = []
        for i in range(k):
            gmeanIni = gmeanlist[i]
            gmeanCros = gmeanlist[i + k]
            if gmeanIni < gmeanCros:
                Bestgmean.append(gmeanCros)
                selectArray.append(CrossArray[i])
            else:
                Bestgmean.append(gmeanIni)
                selectArray.append(InitialArray[i])

        return selectArray, Bestgmean

