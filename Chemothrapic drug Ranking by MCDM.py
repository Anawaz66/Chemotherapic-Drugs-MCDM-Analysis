# Import the necessary libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sympy import symbols, init_printing, Rational, Matrix, sqrt, latex, pprint, simplify
import pandas as pd


import time
# Start the timer
start_time = time.time()
# Initialize pretty printing
init_printing()

#a). Daunorubicin
connections1 = [(1 , 2 ) , (2 , 3 ) ,  (2 , 4 ) , (4 , 5 ) , (4 , 6 ) , (4 , 18 ) , (6 , 7 ) , (7 , 8 ) , (7 , 20),
(8 , 9) , (9 , 10 ), (9 , 17 ) , (10 , 11) , (11 , 12 ) , (11 , 13 ) , (13 , 14 ) , (13 , 15) , (15 , 16) , (15 , 17),
(18 , 19) , (19 , 20) , (19 ,21) , (20, 25) , (21 , 22) , (21 , 23) , (23 , 24) , (23 ,27) , (24 , 25) , (24 , 31) ,
(25 , 26) , (27, 28) , (27 , 29) , (29 , 30) , (29 , 33) , (30 , 31) , (30 , 36) , (31 , 32) , (33 , 34) , (34 , 35) ,
(35 , 36) , (36 , 37) , (37 , 38)]
# b). Dacarbazine
connections2 = [(1 , 2) , (2 , 3) , (2 , 4) , (4 , 5) , (5 , 6) , (6 , 7) , (6 , 10) , (7 , 8) , (8 , 9) , (9 , 10),
(10 , 11) , (11 , 12) , (11 , 13)]
# c). Bendamustine
connections3 = [(1 , 2) , (2 , 3) , (3 , 4) , (4 , 5) , (4 , 8) , (5 , 6) , (6 , 7) , (8 , 9) , (8 , 13) , (9 , 10),
(10 , 11) , (11 , 12) , (11 ,14 ) , (12 , 13) , (12 , 17) , (14 , 15) , (14 , 16) , (16 , 17) , (16 , 18) , (18 , 19),
(19 , 20) , (20 , 21) , (21 , 22) , (21 , 23)]
# d). Dactinomycin
connections4 = [(1 , 2), (2 , 3) , (2 , 4) , (4 , 5) , (4 , 34) , (5 , 6) , (5 , 7) , (7 , 8), (7 , 11), (8 , 9), (9, 10),
(10 , 11) , (11,  12), (11 , 13) , (13 , 14) , (13 , 15) , (15 , 16) , (15 , 17) , (17 , 18), (18 , 19) , (18, 20), (20, 21),
(20 , 22) , (22 , 23), (22 , 26) , (23 , 24) , (23 , 25) , (26 , 27) , (26 , 28) , (28 , 29), (29, 31) , (31, 32),
(31 , 35) , (32 , 33), (32 , 34) , (35 , 36) , (36 , 37) , (36 , 38) , (38 , 39) , (38 , 44) , (39 , 40) , (40, 41), (41, 42),
(41 , 43) , (43 , 44), (43 , 48) , (44 , 45) , (45 , 46) , (46 , 47) , (46 , 55) , (47 , 48) , (47 , 50) , (49, 50), (50, 51),
(51 , 52) , (51 , 53), (53 , 54) , (53 , 55) , (55 , 56) , (56 , 57) , (56 , 58) , ( 59 , 60) , (59 , 87), (60, 61), (60, 62),
(62 , 63) , (63 , 64), (63 , 67) , (64 , 65) , (64 , 66) , (67 , 68) , (67 , 69) , (69 , 70) , (69 , 73) , (70, 71), (71, 72),
(72 , 73) , (73 , 74), (74 , 75) , (74 , 76) , (76 , 77) , (76 , 78) , (78 , 79) , (79 , 80) , (79 , 81) , (81, 82), (81, 83),
(83 , 84) , (83 , 89), (84 , 85) , (84 , 86) , (86 , 87) , (87 , 88) , (89 , 90) , ( 89 , 91),(29 , 30), (57 , 59), (73,92)]
# (29 , 30), (57 , 59), (73,92)  dotted
# e). Ifosfamide
connections5 = [(1 , 2 ) , (2 , 3) , (3 , 4) ,  (4 , 5) , (5 , 6) , (5 , 7) , (5 , 11) , (7 , 8) , (8 , 9) , (9 , 10) , (10 , 11) , (11 , 12) , (12 , 13) , (13 , 14)]
# f). Methotrexate
connections6 = [(1, 2), (2, 3) , (3 , 4) , (4 , 5) , (5 , 6) , (6 , 7), (6 , 8), (2 , 8),(4, 9), (9,10), (10, 11),
(11, 12), (12,5), (11 ,13), (13,14),(14 , 15), (14, 16), (16, 17),(16, 18), (17, 19), (19, 20), (20, 21),(21 ,18),
(19 , 22), (22,23), (22,24), (24,25), (25,26), (26,27), (27,28), (28,29), (28,30),(25,31), (31,32),(31,33)]
# g). Doxorubicin
connections7 = [(1 , 2) , (2 , 3) , (3 , 4) , (3 , 5) , (5 , 6) , (5 , 7) , (5 , 11) , (7 , 8) , (8 , 9) , (8 , 30) , (9 , 10) , (9 , 16) , (10 , 11),  (10 , 12 ) , (12 , 13) , (12 , 14) , (14 , 15) , (14 , 18) , (15 , 22) , (15 , 16) , (16 , 17) , (18 ,19) , (18 , 20) , (20 ,21) , (20 , 24) , (21 , 22) , (21 , 27) , (22 , 23) , (24 , 25), (25 , 26) , (26 , 27) , (27 , 28) , (28 , 29) , (30 , 31), (31 , 32) , (31 , 39) , (32 , 33) , (33 , 34) , (33 , 35) , (35 ,36) , (35 , 37) , (37 , 38) , (37 , 39)]
# h). Capecitabine
connections8 = [(1 , 2),  (2 , 3),  (3 , 4), (4 , 5) , (5 , 6) , (6 , 30), (30 , 7), (30 , 31),(31,8),(31,9),
(9 , 10),  (9 , 12), (10 , 13) , (11 , 12),  (12 , 16), (13 , 14) , (13 , 15),  (15 , 16), (15, 17) , (17 , 18), (17 , 19) , (18 , 20) , (20 , 21),  (20 ,22 ), (22, 23), (23, 19), (19, 24)]

# I). Cabazitaxel DB06772
connections9 = [(1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (6, 7), (6, 8), (8, 9), (9, 10), (9, 16), (10, 11), (10, 15), (11, 12 ), (12, 13),
(13 , 14) , (14 , 15) , (16 , 17) , (16 , 18) , (18 , 19) , (18 , 20) , (20 , 21) , (21 , 22) , (21 , 30) ,  (22 , 23) , (22 , 24) , (24 , 25) , (24 , 31) , (25 , 26) , (25 , 27) , (25 , 28) , (28 , 29) , (28 , 30) , (28 , 40) , (31 , 32) , (31 , 34) , (32 , 33) , (34 , 35) , (34 , 36) , (36 , 37 ) , (36 , 38) , (36 , 50) , (38 , 39) ,
(38, 40), (38, 58), (40, 41), (41, 42), (42, 43), (42, 44), (44, 45),(44, 49), (45, 46), (46, 47), (47, 48), (48, 49), (50, 51), (50, 53),
(51, 52), (53, 54),(54, 55), (54, 56), (54, 58), (56, 57), (57, 58), (58, 59) , (59, 60), (60 , 61), (60 , 62)]
# J). Cyclophosphamide  DB00531
connections10 = [(1 , 2) , (2 , 3) , (3 , 4) , (4 , 5) , (4 , 9) , (5 , 6), (6 , 7), (8 , 9) , (8 , 14) , (9 , 10) , (9 , 11) , (11 , 12 ) , (12 , 13) ,  (13 , 14)]

# 11).  Sunitinib  DB01268
connections11 = [(1, 2), (2 , 3), (3 , 4),(4 , 5),(5 , 6), (6,7),(6,8),(8,9),(9,10), (10,2),(8,11),(11,12),(12,13), (13,14),(14,15),(14,16),(16,17),(17,18),(17,12),(16,19),(19,20),(20,21),(20,22),
                 (22,23),(23,24),(24,25),(25,26),(26,27),(25,28),(25,29)]
# 12). paclitaxel
connections12 = [(1,2), (2,3), (3,4), (4,5), (5,6),(6,1), (4,7), (7,8), (7,9), (9,10),(10,11), (11,12), (12,13), (13,14), (14,15),(15,16), (16,11),
                 (10,17), (17,18), (17,19), (19,20), (19,21), (21,22), (22,23), (23,24),(23,25), (25,26), (26,27), (27,28), (28,29),(28,30), (26,31), (31,32), (32,33), (31,34),
                 (33,35), (35,36), (36,37), (37,38), (38,39), (38,40), (40,41),(40,42), (38,42), (42,43), (43,45),
                 (43,44), (44,45), (45,46), (45,47), (47,48),(48,49), (49,50),(50,51), (51,52), (52,47), (45,51),(51,52), (51,37), (51,25), (45,54), (54,22),]

# 13) Imatinib (DB00619) - 1-indexed
connections13 = [
    (1,2), (1,3), (2,4), (4,6), (3,5), (5,6), (6,7), (7,8), (8,9), (9,10),
    (10,11), (11,12), (12,7), (11,13), (13,14), (14,15), (15,16), (15,17),
    (17,18), (18,19), (19,20), (20,14), (19,21), (21,22), (22,23), (22,24),
    (24,25), (25,26), (26,27), (27,28), (28,29), (29,24), (27,30), (30,31),
    (31,32), (32,33), (33,34), (34,35), (34,36), (36,37), (37,38), (38,31)
]
# 14) Adagrasib (DB15568) - 1-indexed
connections14 = [
    (1,2), (2,3), (3,4), (4,5), (4,6), (6,7), (7,1), (7,8), (8,9), (9,10),
    (10,11), (11,12), (11,6), (12,13), (13,14), (14,15), (15,16), (16,17),
    (17,12), (15,18), (18,19), (18,20), (20,21), (21,22), (22,23), (23,24),
    (24,25), (24,26), (26,27), (26,28), (29,30), (30,31), (31,32), (30,23),
    (19,33), (33,34), (34,35), (35,36), (36,37), (37,38), (38,39), (39,40),
    (40,41), (41,36)
]

# 15) Afatinib (DB08916) - 1-indexed
connections15 = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (9,11),
    (11,12), (11,13), (13,7), (4,14), (14,15), (15,1), (15,16), (16,17),
    (17,18), (18,19), (19,20), (20,21), (21,22), (22,23), (23,19), (17,24),
    (24,25), (25,14), (24,26), (26,27), (27,28), (27,29), (29,30), (30,31),
    (31,32), (32,33), (32,34)]



# Docetaxel
connections_docetaxel = [(1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 16), (16, 17), (16, 18), (18, 19), (19, 20), (19, 21), (21, 22), (22, 23), (23, 24), (23, 25), (25, 26), (25, 27), (27, 28), (28, 29), (29, 30), (29, 31), (31, 32), (32, 33), (32, 34), (32, 35), (27, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (21, 42), (42, 43), (43, 44), (43, 45), (45, 46), (46, 47), (47, 48), (47, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (45, 55), (43, 56), (56, 57), (56, 58), (8, 5), (55, 12), (55, 5), (56, 18), (41, 36), (54, 49)]

# Gemcitabine
connections_gemcitabine = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (11, 13), (13, 14), (13, 15), (5, 16), (16, 17), (16, 18), (18, 2), (13, 6)]




G = nx.Graph()
G.add_edges_from(connections15)
num_vertices = G.number_of_nodes()
print("The number of vertices in the graph is:", num_vertices)
num_edges = G.number_of_edges()
print("The number of edges in the graph is:", num_edges)

#Visualize the graph
pos = nx.kamada_kawai_layout(G)   # Position the nodes using a spring layout
plt.figure(figsize=(12, 14))
# plt.axis('off')
nx.draw(G, pos, with_labels=True, node_size=110, width=1.5,  node_color='lightblue', font_size=11)
plt.title("Custom Graph")
plt.show()

# 1 Wiener Index
wiener_index = sum(nx.shortest_path_length(G, source=u, target=v) for u in G.nodes() for v in G.nodes() if u != v) / 2
print(f"Wiener Index: {wiener_index}")

# 2 , 3
zagrebco_m1 = sum(G.degree(u) + G.degree(v) for u, v in nx.non_edges(G))
zagrebco_m2 = M2 = sum(G.degree(u) * G.degree(v) for u, v in nx.non_edges(G))
print(f"First Zagreb Co Index: {zagrebco_m1}")
print(f"Second Zagreb Co Index: {zagrebco_m2}")
# 4
Forgotten_topological_index = sum(G.degree(u)**2 + G.degree(v)**2 for u, v in G.edges())
Forgotten = Forgotten_topological_index
print(f"Forgotten_topological_index (FT): {Forgotten}")

# 5 Corrected Degree Distance Index
degree_distance_index = sum(d * sum(nx.shortest_path_length(G, source=v).values()) for v, d in G.degree())
print(f"Degree Distance Index: {degree_distance_index}")

# 6 Gutman Index
gutman_index = sum(G.degree(u) * G.degree(v) * nx.shortest_path_length(G, source=u, target=v) for u in G.nodes() for v in G.nodes() if u != v)
print(f"Gutman Index: {gutman_index}")

# 7 Eccentric Distance Sum
eccentricities = nx.eccentricity(G)
# Compute distances for each vertex
distances = {}
for v in G.nodes():
    distances[v] = sum(nx.single_source_shortest_path_length(G, v).values())

# Calculate EDS
EDS = sum(eccentricities[v] * distances[v] for v in G.nodes())
print(f"Eccentric Distance Sum (EDS): {EDS:.3f}")
# 8 total eccentricity index
TEI = sum(eccentricities[v]  for v in G.nodes())
print(f"Total eccentricity index  (TEI): {TEI:.3f}")

import numpy as np

# after computing all indices
results = [wiener_index, zagrebco_m1, zagrebco_m2, Forgotten, degree_distance_index, gutman_index, EDS, TEI]

# convert to NumPy row vector
row_matrix = np.array([results])
print("Row matrix of indices:")
print(row_matrix)



# D = np.array([ [4100, 2892, 3150, 606, 17328, 36472, 92628, 416],
# [262, 252, 242, 152, 982, 1816, 3128, 75], [1313, 944, 967, 282, 5180, 10134, 27478, 230],
# [47828, 17340, 18355, 1384, 198814, 412776, 2163428, 2028], [307, 300, 288, 166, 1150, 2128, 3883, 85],
# [3650, 2072, 2173, 436, 14999, 30670, 104752, 457], [4392, 3048, 3313, 614, 18507, 38834, 99740, 429],
# [1821, 1222, 1246, 330, 7234, 14286, 43445, 299],])
# [14444, 7824, 8363, 1040, 59956, 124074, 425824, 888],
# [301, 300, 288, 166, 1132, 2104, 3478, 78]
# ])

# Option 1: Specify dtype directly (or)
data = np.array([
    [4100, 2892, 3150, 606, 17328, 36472, 92628, 416],
    [262, 252, 242, 152, 982, 1816, 3128, 75],
    [1313, 944, 967, 282, 5180, 10134, 27478, 230],
    [47828, 17340, 18355, 1384, 198814, 412776, 2163428, 2028],
    [307, 300, 288, 166, 1150, 2128, 3883, 85],
    [3650, 2072, 2173, 436, 14999, 30670, 104752, 457],
    [4392, 3048, 3313, 614, 18507, 38834, 99740, 429],
    [1821, 1222, 1246, 330, 7234, 14286, 43445, 299],
    [301, 300, 288, 166, 1132, 2104, 3478, 78],
    [2572, 1536, 1565, 384, 10241, 20302, 66286, 360],
    [9586, 6012, 6828, 1058, 42521, 93932, 257167, 698],
    [5735, 2910, 3201, 492, 24976, 54346, 200318, 640],
    [6456, 3314, 3530, 514, 27084, 56644, 182158, 560],
    [3732, 2266, 2447, 446, 15700, 32906, 96169, 421],
    [14444, 7824, 8363, 1040, 59956, 124074, 425824, 888]

], dtype=np.int64)
print("Table 2: Decision Matrix (Data):")
print(data)

print("Entropy details:")
col_sums = data.sum(axis=0)
# Create the normalized decision matrix N
N = data / col_sums
np.set_printoptions(precision=5, suppress=True)
print("\nTable 3: Normalized Decision Matrix (N, entropy):")
print(N)

# Calculate entropy for each result
e_values = -1 / np.log(15) * np.sum(N * np.log(N), axis=1)

# Calculate weights
weights = (1 - e_values) / np.sum(1 - e_values)

print("Entropy values:")
print(e_values)
print("Divergence d_j:")
print(1 - e_values)
print("\nAttribute Weights:")
print(weights)
w = np.sum(weights)
print(w)


print("TOPSIS Method")
# ("Table 2 as initial")
# Compute square sum column-wise
col_sum_sqr = np.sum(data**2, axis=0)
# Compute square root column-wise
col_sum_sqrt = np.sqrt(col_sum_sqr)
print("Square Sum:", col_sum_sqr)
print("Square Root:", col_sum_sqrt)

# Create the normalized decision matrix N
N1 = data / col_sum_sqrt
np.set_printoptions(precision=5, suppress=True)
print("\nTable 3: Normalized Decision Matrix (N):")
print(N1)

print("Table 4: Weights Matrix:")
# Calculate entropy for each result
e_values1 = -1 / np.log(8) * np.sum(N1 * np.log(N1), axis=1)
print("E_j:")
print(e_values1)
print("d_j values topsis:")
d_j=1 - e_values1
print(d_j)
print("\nWeights TOPSIS:")
# Calculate weights
weight1 = (d_j) / np.sum(d_j)
print(weight1)
w = np.sum(weight1)
print(w)


print("Table 3, 4: as initial")
# W_matrix = N * weight1

# reshape weights to (15,1) so they broadcast across columns
W_matrix = N * weight1[:, None]
# after addition of new data

print("Weighted Normalized Evaluation Matrix (WNM):")
print(W_matrix)

print("Table 5: max, min values:")
is_positive = np.array([True, True, True, True, True, True, False, True])
# Compute the column-wise maximum and minimum values.
col_max = np.max(W_matrix, axis=0)
col_min = np.min(W_matrix, axis=0)

# - Maximum for positive attributes, Minimum for negative attributes and voice versa
ideal_alternative = np.where(is_positive, col_max, col_min)
anti_ideal_alternative = np.where(is_positive, col_min, col_max)

# print("Column-wise Maximums:", col_max)
# print("Column-wise Minimums:", col_min)
print("Ideal Alternative:", ideal_alternative)
print("Anti-Ideal Alternative:", anti_ideal_alternative)

# Distance to the ideal best
# Distance to the ideal worst
distance_positive = np.sqrt(((W_matrix - col_max) ** 2).sum(axis=1))
distance_negative = np.sqrt(((W_matrix - col_min) ** 2).sum(axis=1))
print("distance_positive:", distance_positive)
print("distance_negative:", distance_negative)

# 7. Compute TOPSIS Scores (Relative Closeness Coefficient)
closeness_coeff = distance_negative / (distance_positive + distance_negative)
# print(closeness_coeff)
# Attach the scores to the DataFrame
Oi=np.array(closeness_coeff)
print("Oi:")
print(Oi)
# Sort the values in descending order
sorted_indices = np.argsort(Oi)[::-1]
ranks = {value: rank + 1 for rank, value in enumerate(sorted_indices)}
# Get the ranks for each value in Ui
Oi_ranks = [ranks[i] for i in range(len(Oi))]
Oi_ranks = np.array(Oi_ranks)
print("Table 6:")
# print all values of phi
print(Oi_ranks)
rank_matrix1 = np.hstack((distance_positive.reshape(-1, 1), distance_negative.reshape(-1, 1), closeness_coeff.reshape(-1,1),  Oi_ranks.reshape(-1, 1)))
np.set_printoptions(precision=5, suppress=True)
# Format the values in Rank_matrix as decimals
# Rank_matrix = np.around(rank_matrix, decimals=decimal_precision)
print("Table 7: Topsis Rank Matrix:")
print(rank_matrix1)


print("COPRAS Method:")

print("Table 9, 10: as initial")
W_matrix = N * weights[:, None]
print("Weighted Normalized Evaluation Matrix (WNM):")
print(W_matrix)
# Sum every column
column_sums = np.sum(W_matrix, axis=0)
print("Column Sums:")
print(column_sums)

row_sums = W_matrix[:, [0, 2, 3, 4, 5, 6]].sum(axis=1)
print("Beneficial rows sum")
print(row_sums)
# Create a new matrix with the row sums and the remaining entries
wbnc_matrix = np.hstack((row_sums.reshape(-1, 1), W_matrix[:, [1]]))
print("Table 11: Weighted Benef. non-benf. criteria Matrix:")
print(wbnc_matrix)

#
min_Ci = np.min(wbnc_matrix[:, 1])
sum_Ci = np.sum(wbnc_matrix[:, 1])
# Compute the ratio (min Ci / Ci) for each element
ratios = min_Ci / wbnc_matrix[:, 1]
print("ratios:", ratios)
new_matrix = np.hstack((wbnc_matrix, ratios.reshape(-1, 1)))
# Print the new matrix
print("Table 12: combine matrix:")
print(new_matrix)

# alternative Qi
Qi = row_sums + ((min_Ci * sum_Ci) / (wbnc_matrix[:, 1] * np.sum(ratios)))
Qi = np.array(Qi)
# print(row_sums, min_Ci, sum_Ci, (wbnc_matrix[:, 1]), np.sum(ratios))
print("Qi:")
print(Qi)

Ui = (Qi / np.max(Qi)) * 100
Ui = np.array(Ui)
np.set_printoptions(precision=5, suppress=True)
# Sort the values in descending order
sorted_indices = np.argsort(Ui)[::-1]
# Create a dictionary to store the ranks
ranks = {value: rank + 1 for rank, value in enumerate(sorted_indices)}
# Get the ranks for each value in Ui
Ui_ranks = [ranks[i] for i in range(len(Ui))]
Ui_ranks = np.array(Ui_ranks)

# Create Rank_matrix by horizontally stacking the arrays
rank_matrix = np.hstack((Qi.reshape(-1, 1), Ui.reshape(-1, 1), Ui_ranks.reshape(-1, 1)))
# decimal_precision = 6
np.set_printoptions(precision=5, suppress=True)
print("Table 13: Ui Rank Matrix:")
print(rank_matrix)





# print(type(rank_matrix1))
# df = pd.DataFrame(rank_matrix1)
# df.to_excel("rank_result.xlsx", index=False, header=False)


# # 3d graph
# # Daunorubicin, Dacarbazine, Bendamustine, Dactinomycin, Ifosfamide, Methotrexate, Doxorubicin, Capecitabine
# # Create the data
# data = {
#     'Drugs': ['Daunorubicin', 'Dacarbazine', 'Bendamustine', 'Dactinomycin', 'Ifosfamide', 'Methotrexate', 'Doxorubicin', 'Capecitabine',
#               'Cabazitaxel', 'Cyclophosphamide', 'Sunitinib', 'Paclitaxel', 'Imatinib', 'Adagrasib', 'Afatinib'],
#     # 'Copras': [5, 2, 7, 1, 3, 6, 4, 8],
#     # 'Topsis': [3, 8, 6, 1, 7 ,4, 2, 5]
#     'Copras': [6, 11, 12, 15, 13, 8, 5, 10, 14, 9, 1, 3, 2, 7, 4],
#     'Topsis': [10, 9,  4,  1, 7,  5, 11,  2,  8,  3, 14, 12, 13,  6, 15]
#
# }
#
# # Create DataFrame
# df = pd.DataFrame(data)
# # Setting up the 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Define the positions for bars
# x_pos = np.arange(len(df))
# y_pos = np.array([0, 1])  # Two positions: one for Copras, one for Promethee
#
# # Reduce the bar width
# dx = dy = 0.17  # Reduced width of the bars
# dz = df[['Copras', 'Topsis']].values
#
# # Assign one color per drug
# colors = ['magenta', 'yellow', 'cyan', 'green', 'purple', 'red', 'blue', 'darkblue']
#
# # Plotting the bars with specified colors
# for i in range(len(df)):
#     ax.bar3d(x_pos[i], y_pos[0], 0, dx, dy, dz[i, 0], color=colors[i], shade=False, edgecolor='black')
#     ax.bar3d(x_pos[i], y_pos[1], 0, dx, dy, dz[i, 1], color=colors[i], shade=True, edgecolor='black')
#
# # Set labels and title
# ax.set_xticks(x_pos)
# ax.set_xticklabels(df['Drugs'], rotation=30, ha='right', fontsize=12, fontweight='normal', color='purple')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(['COPRAS', 'TOPSIS'], fontsize=12, fontweight='bold', color='purple')
# ax.set_zlabel('Ranking', fontsize=12, fontweight='bold', color='purple')
#
# # Adjust the view angle
# ax.view_init(elev=40, azim=-50)
#
# # Add title
# ax.set_title('Ranking of Drugs', fontsize=14, fontweight='bold', color='purple', pad=10)
#
# # Show the plot
# plt.show()

# # R Values graph
# # Data in a 2D list or NumPy array
# data = np.array([
#     [-0.309, 0.545, 0.431, 0.958, 0.963, 0.864, 0.358],
#     [-0.319, 0.514, 0.369, 0.923, 0.928, 0.818, 0.354],
#     [-0.286, 0.565, 0.528, 0.984, 0.983, 0.922, 0.290],
#     [-0.300, 0.554, 0.428, 0.933, 0.939, 0.843, 0.404],
#     [-0.309, 0.530, 0.462, 0.976, 0.979, 0.896, 0.280],
#     [-0.299, 0.554, 0.491, 0.981, 0.983, 0.908, 0.306],
#     [-0.309, 0.538, 0.458, 0.977, 0.980, 0.892, 0.303]
# ])
#
# # Number of columns
# columns = ["DS", "E", "FP", "M", "P", "MV", "BP"]
#     # ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]
#
# # Plot each row as a separate line
# plt.figure(figsize=(10, 6))
# for i, col_name in enumerate(columns):
#     plt.plot(columns,data[:, i], linewidth=1.65, marker='o', label=col_name)
#
# # Customize the plot
# # plt.title("Line Plot of Data")
# plt.xlabel("Physicochemical properties")
# plt.ylabel("Correlation coefficient")
# plt.legend(title="Rows",  title_fontsize="12")
# plt.grid(True)
# plt.show()


# Original data
data = {
    'Drugs': ['Daunorubicin', 'Dacarbazine', 'Bendamustine', 'Dactinomycin', 'Ifosfamide', 'Methotrexate', 'Doxorubicin', 'Capecitabine',
              'Cabazitaxel', 'Cyclophosphamide', 'Sunitinib', 'Paclitaxel', 'Imatinib', 'Adagrasib', 'Afatinib'],
    # 'Copras': [5, 2, 7, 1, 3, 6, 4, 8],
    # 'Topsis': [3, 8, 6, 1, 7 ,4, 2, 5]
    'Copras': [6, 11, 12, 15, 13, 8, 5, 10, 14, 9, 1, 3, 2, 7, 4],
    'Topsis': [10, 9,  4,  1, 7,  5, 11,  2,  8,  3, 14, 12, 13,  6, 15]

}
df = pd.DataFrame(data)

# ----- SORTING INSERTED HERE -----
# Sort by COPRAS rank (lower is better)
df_sorted = df.sort_values('Copras').reset_index(drop=True)
# ---------------------------------

drugs = df_sorted['Drugs']
x = np.arange(len(drugs))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, df_sorted['Copras'], width, label='COPRAS', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, df_sorted['Topsis'], width, label='TOPSIS', color='coral', edgecolor='black')

ax.set_xlabel('Drugs(sorted by COPRAS)', fontsize=12, fontweight='bold')
ax.set_ylabel('Rank', fontsize=12, fontweight='bold')
ax.set_title('Ranking of Drugs by COPRAS and TOPSIS', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(drugs, rotation=45, ha='right', fontsize=10)
ax.legend()

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# End the timer
end_time = time.time()
# Calculate the time taken
time_taken = end_time - start_time
print(f"Time taken to generate connections: {time_taken} seconds")
