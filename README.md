# Crowdsourcing VRP

*Dummy Agent*: it returns the delivery in the same order as their arrival

*exactVRPAgent:*

- Decide which delivery to crowdship based on the l2 distance 
- Solve in an exact way the VRP by means of Lazy constraints

# Problem description
Here we have a method in TSP which do not require to visit all customers, along a gradient of ‘‘decision power’’ attributed to the tour planner.At one extreme of this gradient lies the case in which the tour planner has no choice over which customer will require a delivery (we assume that we visit customers to deliver some goods) and which will not, because a random variable determines customer presence. This problem is the Probabilistic Travelling Salesman Problem (PTSP). In the PTSP a decision-maker has to devise a tour to visit a set of delivery points, some of which might be later revealed not to be available. Because the decision-maker does not know in advance which customers will drop out, he/she faces two options:

- The first option is to solve a TSP problem for each possible set of delivery points,wait until the status of all customers is revealed and use the TSP tour visiting the customers requiring delivery. This strategy is computationally expensive.
- The second one which calls a priori approach. In this approach, in first planning an a priori tour visiting all the customers. When the stochastic outcome is revealed, the decision-maker amends the solution, skips the deliveries that are not required, and performs the remaining ones in the same order as they appear in the "a priori tour". This has an advantage that, when the problem is solved for a multi-day planning horizon, all routes will be similar.

Given a graph G=(V, A) in the figure below. Denote with $O \subseteq V$ the subset of deliveries offered for crowdsourcing and with $A \subseteq O$ the set of accepted offers, which is only revealed at the end of the day.

<figure>
  <img src="/Image/General_Idea.png" alt="Example of a graph">
  <figcaption>Figure 1: The relation between sets, V, O and A. The figure also shows the TSP tour of the owned vehicle when A is the deliveries accepted for crowdsourcing</figcaption>
</figure>

 In this sense, the problem is a two-stage problem and the set of accepted offers is only revealed in the second stage. The decision-maker has to decide which deliveries to offer for crowdsourcing.
