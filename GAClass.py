# -*- coding: utf-8 -*-
"""
	遗传算法GA包
	Author:	Greatpan
	Date:	2018.11.13
"""
import random

class GAUnit(object):
	"""
		类名：GAUnit
		类说明：	遗传算法个体类
	"""
	def __init__(self, _gene = None, _score = -1):
		""" 构造函数 """
		self.gene = _gene			# 个体的基因序列
		self.value = _score  	# 初始化适配值


class GAList(object):
	"""
		类名：GAList
		类说明：	遗传算法类,一个GA类包含了一个种群,及其种群内的信息
	"""
	def __init__(self, _cross_rate, _mutation_rate, _unit_amount, _gene_length, _MatchFun=lambda : 1):
		""" 构造函数 """
		self.cross_rate = _cross_rate  		    # 交叉概率 #
		self.mutation_rate = _mutation_rate  	# 突变概率 #
		self.unit_amount = _unit_amount   		# 个体数 #
		self.gene_length = _gene_length  		# 基因长度 #
		self.MatchFun = _MatchFun  			    # 适配函数
		self.population = []  	 # 种群
		self.best = None  		 # 保存这一代中最好的个体
		self.generation = 1  	 # 第几代 #
		self.cross_count = 0  	 # 交叉数量 #
		self.mutation_count = 0  # 突变个数 #
		self.bounds = 0.0  		 # 适配值之和，用于选择时计算概率
		self.initPopulation()  	 # 初始化种群 #

	def initPopulation(self):
		"""
			函数名：initPopulation(self)
			函数功能：	随机初始化得到一个种群
				输入	1 	self：类自身
				输出	1	无
			其他说明：无
		"""
		self.population = []
		unit_amount = self.unit_amount
		while unit_amount>0:
			zero_amt = self.gene_length//2
			one_amt = self.gene_length - zero_amt
			gene = ['0'] * zero_amt + ['1'] * one_amt
			random.shuffle(gene)  			# 随机洗牌 #
			gene = ''.join(gene)
			unit = GAUnit(gene)
			self.population.append(unit)
			unit_amount -= 1

	def judge(self):
		"""
			函数名：judge(self)
			函数功能：	重新计算每一个个体单元的适配值
				输入	1 	self：类自身
				输出	1	无
			其他说明：无
		"""
		self.bounds = 0.0
		self.best = self.population[0]
		for unit in self.population:
			unit.value = self.MatchFun(unit)
			self.bounds += abs(unit.value)
			if self.best.value < unit.value:	# sharpe ratio 越大越好 #
				self.best = unit

	def cross(self, parent1, parent2):
		"""
			函数名：cross(self, parent1, parent2)
			函数功能：	根据parent1和parent2基于序列,随机选取长度为n的片段进行交换(n=index2-index1)
				输入	1 	self：类自身
					2	parent1: 进行交叉的双亲1
					3	parent2: 进行交叉的双亲2
				输出	1	newGene： 通过交叉后的一个新的遗传个体的基因序列号
			其他说明：进行交叉时采用的策略是,将parent2的基因段tempGene保存下来,然后对基因1所有序列号g依次进行判断,
				如果g在tempGene内,则舍去,否则就保存下来,并在第index1的位置插入tempGene
		"""
		index1 = random.randint(0, self.gene_length - 1)  		# 随机生成突变起始位置 #
		index2 = random.randint(index1, self.gene_length - 1)  	# 随机生成突变终止位置 #
		new_gene = parent1.gene[0:index1] + parent2.gene[index1:index2] + parent1.gene[index2:]
		self.cross_count += 1
		return new_gene

	def mutation(self, gene):
		"""
			函数名：mutation(self, gene)
			函数功能：	对输入的gene个体进行变异,也就是随机交换两个位置的基因号
				输入	1 	self：类自身
					2	gene: 进行变异的个体基因序列号
				输出	1	newGene： 通过交叉后的一个新的遗传个体的基因序列
			其他说明：无
		"""
		new_gene = list(gene[:])  					# 产生一个新的基因序列，以免变异的时候影响父种群
		for i in range(5):
			index1 = random.randint(0, self.gene_length - 1)
			index2 = random.randint(0, self.gene_length - 1)
			# 随机选择两个位置的基因交换--变异
			new_gene[index1], new_gene[index2] = new_gene[index2], new_gene[index1]
		self.mutation_count += 1
		return ''.join(new_gene)

	def getOneUnit(self):
		"""
			函数名：getOneUnit(self)
			函数功能：	通过轮盘赌法,依据个体适应度大小,随机选择一个个体
				输入	1 	self：类自身
				输出	1	unit：所选择的个体
			其他说明：无
		"""
		r = random.uniform(0, self.bounds)
		for unit in self.population:
			r -= abs(unit.value)
			if r <= 0:
				return unit

		raise Exception("选择错误", self.bounds)

	def newChild(self):
		"""
			函数名：newChild(self)
			函数功能：	按照预定的概率进行交叉与变异后产生新的后代
				输入	1 	self：类自身
				输出	1	GAUnit(gene)：所产生的后代
			其他说明：无
		"""
		parent1 = self.getOneUnit()
		rate = random.random()

		# 按概率交叉
		if rate < self.cross_rate:	# 交叉
			parent2 = self.getOneUnit()
			gene = self.cross(parent1, parent2)
		else:						# 不交叉
			gene = parent1.gene

		# 按概率突变
		rate = random.random()
		if rate < self.mutation_rate:
			gene = self.mutation(gene)

		return GAUnit(gene)

	def nextGeneration(self):
		"""
			函数名：nextGeneration(self)
			函数功能：	产生下一代
				输入	1 	self：类自身
				输出	1	无
			其他说明：无
		"""
		self.judge()
		new_population = []						# 新种群
		new_population.append(self.best)  		# 把最好的个体加入下一代 #
		while len(new_population) < self.unit_amount:
			new_population.append(self.newChild())
		self.population = new_population
		self.generation += 1

