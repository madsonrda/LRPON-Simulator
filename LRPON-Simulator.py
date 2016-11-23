#!/usr/bin/env python
# -*- coding: utf-8 -*-

import simpy
import random
import time
import simpy.rt
import itertools #Depois ver
from operator import itemgetter, attrgetter
import sys
CONTAINER_SIZE = 10
MAX_BUFFER = 625
RANDOM_SEED = int(sys.argv[1])
# CONSIDERANDO 210000 KM/S DE TRANSMISSÃO
class OLT:
	def __init__(self, env):
		self.lambda1 = simpy.Container(env, CONTAINER_SIZE, init = CONTAINER_SIZE)
		self.lambda2 = simpy.Container(env, CONTAINER_SIZE, init = CONTAINER_SIZE)
		self.lambda3 = simpy.Container(env, CONTAINER_SIZE, init = CONTAINER_SIZE)
		self.lambda4 = simpy.Container(env, CONTAINER_SIZE, init = CONTAINER_SIZE)
		self.pool = 0
		self.listaA = []
		self.listaB = []
		self.env = env
		self.disperdicio = 0
		#yield env.process(self.Checking(env))

	def	Checking(self,env):
	#	while True:
			tempo = time.clock()
			if self.lambda1.level < 1:
				print ('Comprimento de onda 1 sendo restabelecido  %f' % tempo)
				yield self.lambda1.put(10-self.lambda1.level)
			if self.lambda2.level < 1:
				print ('Comprimento de onda 2 sendo restabelecido  %f' % tempo)
				yield self.lambda2.put(10-self.lambda2.level)
			if self.lambda3.level < 1:
				print ('Comprimento de onda 3 sendo restabelecido  %f' % tempo)
				yield self.lambda3.put(10-self.lambda3.level)
			if self.lambda4.level < 1:
				print ('Comprimento de onda 4 sendo restabelecido  %f' % tempo)
				yield self.lambda4.put(10-self.lambda4.level)

	def Calculo_Janelas(self, atraso, carga):
		bits = carga * 8000000
		tempo_transferencia = 	bits/float(1000000000)
		tempo_total = 	atraso + float(tempo_transferencia)
		return tempo_total

	def Scheduling_Container(self, janela):
		if self.lambda1.level >= janela:
			return self.lambda1
		if self.lambda2.level >= janela:
			return self.lambda2
		if self.lambda3.level >= janela:
			return self.lambda3
		if self.lambda4.level >= janela:
			return self.lambda4

	def DBA(self, env):

		while (len(self.listaA) != 0):
			Windows = self.Calculo_Janelas(self.listaA[0].onu.atraso,self.listaA[0].carga)
			Restante = Windows
			yield env.process(self.Checking(env))
			self.pool = 0
			while (Restante > 0):
				if Restante > 1:
					yield self.Scheduling_Container(1).get(1)
					Restante -= 1
					yield env.timeout(1)
					tempo2 = time.clock()
					tempo3 = tempo2 - self.listaA[0].tempo
					print ('Onu: %s. Thread : %s. Status: Alocada com Atraso %.6fs' % (self.listaA[0].onu.key,self.listaA[0].thread_number,tempo3))
					if self.listaA[0].onu.thread_atraso < tempo3:
						self.listaA[0].onu.thread_atraso = tempo3
				else:
					tempo2 = time.clock()
					tempo3 = tempo2 - self.listaA[0].tempo
					yield self.Scheduling_Container(Restante).get(Restante)
					yield env.timeout(Restante)
					print (' Onu: %d. Thread : %d. Status: Alocada com atraso %.6fs' % (self.listaA[0].onu.key,self.listaA[0].thread_number,tempo3))
					if self.listaA[0].onu.thread_atraso < tempo3:
						self.listaA[0].onu.thread_atraso = tempo3
					self.listaA.pop(0)
					self.pool =+ 1 - Restante
					Restante = 0
					yield env.process(self.DBA2(env))
			self.disperdicio += self.pool
			print ('--------------Desperdício atual largura de banda--------------- ')
			print (' Pool = %.6fs' % self.disperdicio)
			print ('------------------------------------------------------------------------')
		yield env.process(self.DBA3(env))
	#	----------------------------------
		print ('--------------Desperdício acumulado de largura de banda--------------- ')
		print (' Pool = %.6fs' % self.disperdicio)
		print ('------------------------------------------------------------------------')

	def DBA2(self,env):
		#print('Entrei no dba2')
		flag = True

		while (flag):
			index = -1
			Restante = 0
			#print('Tamanho da Lista B: %d' % len(self.listaB))
			i = 0
			while (i < len(self.listaB)):
		#		print ('i = %d' % i)
				Restante = self.Calculo_Janelas(self.listaB[i].onu.atraso,self.listaB[i].carga)
				if  Restante <= self.pool:
					index = i
					break
				i +=1
			if (index != -1):
				tempo2 = time.clock()
				tempo3 = tempo2 - self.listaB[index].tempo
				yield self.Scheduling_Container(Restante).get(Restante)
				yield env.timeout(Restante)
				print (' Onu: %d. Thread : %d. Status: Alocada com atraso %.6fs' % (self.listaB[index].onu.key,self.listaB[index].thread_number,tempo3))
				self.pool -= Restante
				if self.listaB[index].onu.thread_atraso < tempo3:
						self.listaB[index].onu.thread_atraso = tempo3
				self.listaB.pop(index)

			else:
				flag = False

	def DBA3(self,env):
		while(len(self.listaB) != 0):
			Windows = self.Calculo_Janelas(self.listaB[0].onu.atraso,self.listaB[0].carga)
			Restante = Windows
			yield env.process(self.Checking(env))
			self.pool = 0
			tempo2 = time.clock()
			tempo3 = tempo2 - self.listaB[0].tempo
			print (' Onu: %d. Thread : %d. Status: Alocada com atraso %.6fs' % (self.listaB[0].onu.key,self.listaB[0].thread_number,tempo3))
			yield self.Scheduling_Container(Restante).get(Restante)
			yield env.timeout(Restante)
			self.pool = 1-Restante
			self.DBA2(env)
			self.disperdicio += self.pool
			if self.listaB[0].onu.thread_atraso < tempo3:
				self.listaB[0].onu.thread_atraso = tempo3
			self.listaB.pop(0)



####################################Estatística######################################################

class Estatistica:
	def __init__(self,env,onus):
		self.lista = []
		self.media = 0
		self.desvio_padrao = 0
		self.onus = onus
		self.erro_padrao = 0
		self.Analise(env)
		self.Media(env)
		self.DesvioPadrao(env)
		self.ErroPadrao(env)


	def Analise(self,env):
		self.lista = (random.sample(self.onus,50))
		for i in range(0,50):
			print (self.lista[i].thread_atraso)

	def Media(self,env):
		soma = 0
		for i in range (0,50):
			soma += self.lista[i].thread_atraso
		self.media = float(soma/50)

	def DesvioPadrao(self,env):
		soma = 0
		for i in range(0,50):
			soma += pow(self.lista[i].thread_atraso - self.media,2)
		# n - 1 = 49
		div = float(soma/49)
		self.desvio_padrao = pow(div,0.5)

	def ErroPadrao(self,env):
		self.erro_padrao = float(self.desvio_padrao/pow(50,0.5))


########################################################################################




class ONU:
	def __init__(self,distancia,key):
		self.distancia = distancia
		self.key = key
		self.atraso = self.distancia/ float(210000)
		self.thread_atraso = 0


def Setup(env):

	olt = OLT(env)
	onus = []
	j = 0
	for i in range(384):
		distancia = random.randint(20,100)
		onu = ONU(distancia,i)
		onus.append(onu)
	contador = 0
	while contador < 1:
	#	yield env.timeout(5*1000)
		starting = time.clock()
		while j < 384:
		#	ending = time.clock()
		#	ciclo = ending -starting
			random_thread = int(sys.argv[2])
			for k in range(random_thread):
				#random_buffer = random.randint(int(sys.argv[3]),int(sys.argv[4]))
				random_buffer = random.expovariate(0.0)
				tempo1 = time.clock()
				requisicao = Requisicao(env,onus[j],k,random_buffer,tempo1)
				if olt.Calculo_Janelas(onus[j].atraso,random_buffer) >= 1 :
					olt.listaA.append(requisicao)
				else:
					olt.listaB.append(requisicao)

			j += 1
			#if ciclo > 0.005:
			#	break
		contador += 1
		yield env.process(olt.DBA(env))
		print('--------------------------------------------------')
		print('------------------------Resultados estatísticos--------------------------')
		print('Pior tempo de Threads por onu em aleatorio')
		est = Estatistica(env,onus)
		print ('------------------------Média de tempo Amostral-------------------------')
		print ('Media: %f ' % est.media)
		print ('-------------------------Desvio Padrão Amostral--------------------------')
		print ('Desvio padrão = %f' % est.desvio_padrao)
		print ('--------------------------Erro Padrão Amostral --------------------------')
		print ('Erro Padrão = %f' % est.erro_padrao)

#PQ dessa classe?
class Requisicao:

	def __init__(self,env,onu,thread,carga,tempo):
		self.onu = onu
		self.thread_number = thread
		self.carga = carga
		self.tempo = tempo




random.seed(RANDOM_SEED)

env = simpy.rt.RealtimeEnvironment(initial_time=0, factor=0.001, strict=False)
proc = env.process(Setup(env))

env.run(until=proc)
