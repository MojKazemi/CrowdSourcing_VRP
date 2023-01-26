# Crowdsourcing VRP
This project was defined by Prof. Edoardo Fadda in "Operational Research: Theory and Applications" course at 2022 in Politecnico do Torino university
  
# Problem description
Here we have a method in TSP which do not require to visit all customers, along a gradient of ‘‘decision power’’ attributed to the tour planner.At one extreme of this gradient lies the case in which the tour planner has no choice over which customer will require a delivery (we assume that we visit customers to deliver some goods) and which will not, because a random variable determines customer presence. This problem is the Probabilistic Travelling Salesman Problem (PTSP). In the PTSP a decision-maker has to devise a tour to visit a set of delivery points, some of which might be later revealed not to be available. Because the decision-maker does not know in advance which customers will drop out, he/she faces two options:

- The first option is to solve a TSP problem for each possible set of delivery points,wait until the status of all customers is revealed and use the TSP tour visiting the customers requiring delivery. This strategy is computationally expensive.
- The second one which calls a priori approach. In this approach, in first planning an a priori tour visiting all the customers. When the stochastic outcome is revealed, the decision-maker amends the solution, skips the deliveries that are not required, and performs the remaining ones in the same order as they appear in the "a priori tour". This has an advantage that, when the problem is solved for a multi-day planning horizon, all routes will be similar.

Given a graph G=(V, A) in the figure 1. Denote with $O \subseteq V$ the subset of deliveries offered for crowdsourcing and with $A \subseteq O$ the set of accepted offers, which is only revealed at the end of the day.

<figure id="fig:General_idea">
  <p align="center">
    <img src="/Image/General_idea.png" >
  </p>
  <figcaption>Figure 1: The relation between sets, V, O and A. The figure also shows the TSP tour of the owned vehicle when A is the deliveries accepted for crowdsourcing</figcaption>
</figure>

 In this sense, the problem is a two-stage problem and the set of accepted offers is only revealed in the second stage. The decision-maker has to decide which deliveries to offer for crowdsourcing.

# Mathematical Model

We can calculate, for a fixed set O of deliveries offered for crowdsourcing, what is the expected cost *E*<sub>*A*</sub>\[*C*(*O*)\] over all possible realisations of A.  
𝔼<sub>*A*</sub>\[*C*(*O*)\] = ∑<sub>*A* ⊆ *O*</sub>\[(∏<sub>*i* ∈ *A*</sub>*P*<sub>*i*</sub>∏<sub>*i* ∈ *O* ∖ *A*</sub>(1−*P*<sub>*i*</sub>)).(∑<sub>*i* ∈ *A*</sub>*m*<sub>*i*</sub>+*c*<sub>*V*′ ∖ *A*</sub>)\]
  
The objective of the problem is to find the set *O*<sup>*o**p**t*</sup> which gives the lowest expected cost:  
$$O^{opt}= \underset{O \subseteq V}{\arg\min} \mathbb{E}\_A\[C(O)\]$$
  
To solve the problem we can try the heuristic strategies to explore the solution space.we propose approximation methods to efficiently estimate the value of 𝔼<sub>*A*</sub>\[*C*(*O*)\]. computing the objective value 𝔼<sub>*A*</sub>\[*C*(*O*)\] a solution O is hard, because one has to evaluate function C(O, A) each set (*A*⊆*O*) and solve a TSP at each evaluation. Our hypothesis, however, is that it is possible to approximate 𝔼\[*C*(*O*)\] well, while evaluating much fewer functions.By this method we speed up the heuristic algorithms which we are using here.

# K-means Clustering

K-means clustering is a type of unsupervised learning, which is used when there are unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find different clusters in the data, which is the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of the K clusters based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:

-   The centroids of the K clusters, which can be used to label new data

-   Labels for the training data (each data point is assigned to a
    single cluster)

K-means either cluster deliveries that are close to each other or by finding the center of groups, it is possible to define the edge points in each cluster. Clustering deliveries accelerate our method. in addition the number of clusters is determined by the number of vehicles. Output of K-means represents in figure 2.

<figure id="fig:kmean_cluster">
  <p align="center">
    <img src="/Image/kmeancluster.jpg" style="height:7cm" />
  </p>
  <figcaption>Figure 2:K-mean clustering for deliveries in 8 group and presents depot and crowdsourcing point</figcaption>
</figure>

# Nearest Neighbour

The first step of the heuristic VRP is solving construct a feasible initial solution for each group of delivery points. Nearest Neighbour is a simple method to find the initial solution. The tour in each group started with the depot. Afterward, select the nearest delivery in the group to the tour and then add the nearest point to the previous delivery to the tour. After picking the final point in the group, add the depot to the tour. In this way, we have an initial cycle for each group of deliveries to use in the second phase of our solution.  
The drawback of this way is the last destination should come back to the depot at a high cost. As a result, the tours in the first phase were considered initial tours for the 2-opt algorithm.
