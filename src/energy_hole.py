import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(42)
np.random.seed(42)
num_nodes = 25 
plt.style.use('default')

G = nx.random_geometric_graph(num_nodes, radius=0.5, seed=42)
pos = nx.get_node_attributes(G, 'pos')


for u, v in G.edges():
    dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) * 10
    G.edges[u, v]['weight'] = round(dist, 1)

center_point = np.array([0.5, 0.5])
danger_radius = 0.35 

trap_nodes = []
for node in G.nodes():
    node_pos = np.array(pos[node])
    dist_to_center = np.linalg.norm(node_pos - center_point)
    
    if dist_to_center < danger_radius:
        G.nodes[node]['battery'] = 20
        trap_nodes.append(node)
    else:
        G.nodes[node]['battery'] = 300 

source = min(G.nodes(), key=lambda n: sum(pos[n]))    
destination = max(G.nodes(), key=lambda n: sum(pos[n])) 


G.nodes[source]['battery'] = 1000
G.nodes[destination]['battery'] = 1000

print(f"Danger Zone Nodes (Center): {trap_nodes}")


def run_simulation(graph, strategy):
    G_sim = graph.copy()
    rounds = 0
    total_distance = 0
    path_history = []
    
    while True:
        current = source
        path = [current]
        round_cost = 0
        
        while current != destination:
            neighbors = list(G_sim.neighbors(current))
            valid_neighbors = [n for n in neighbors if n not in path]
            
            if not valid_neighbors: break

            next_node = None
            
            if strategy == 'Distance Priority':
                # 무조건 직선 거리 짧은 쪽 (중앙 돌파 시도)
                next_node = min(valid_neighbors, key=lambda n: np.linalg.norm(np.array(pos[n]) - np.array(pos[destination])))
                
            elif strategy == 'Battery Priority':
                # 배터리 가중치 극대화 -> 중앙 위험 지역 절대 회피
                def score(n):
                    batt = G_sim.nodes[n]['battery']
                    dist = np.linalg.norm(np.array(pos[n]) - np.array(pos[destination]))
                    return (batt * 10.0) - (dist * 1.0) # 배터리 중요도 10배
                
                next_node = max(valid_neighbors, key=score)
            
            # 이동 비용
            dist = G_sim.edges[current, next_node]['weight']
            energy_consump = dist * 1.0 
            
            if G_sim.nodes[next_node]['battery'] <= energy_consump:
                return rounds, total_distance, path_history, G_sim
            
            G_sim.nodes[next_node]['battery'] -= energy_consump
            round_cost += dist
            current = next_node
            path.append(current)
        
        if current == destination:
            rounds += 1
            total_distance += round_cost
            path_history = path
        else:
            break
            
    return rounds, total_distance, path_history, G_sim

rounds_dist, dist_dist, path_dist, G_final_dist = run_simulation(G, 'Distance Priority')
rounds_batt, dist_batt, path_batt, G_final_batt = run_simulation(G, 'Battery Priority')

avg_dist_dist = dist_dist / rounds_dist if rounds_dist > 0 else 0
avg_dist_batt = dist_batt / rounds_batt if rounds_batt > 0 else 0


fig = plt.figure(figsize=(16, 8))
plt.subplots_adjust(wspace=0.3)

ax1 = plt.subplot(1, 2, 1)

circle = plt.Circle(center_point, danger_radius, color='orange', alpha=0.1)
ax1.add_patch(circle)
ax1.text(0.5, 0.5, "Low Battery Zone\n(Danger)", horizontalalignment='center', verticalalignment='center', color='darkorange', fontweight='bold')

node_colors = ['orange' if n in trap_nodes else '#76c7c0' for n in G.nodes()]

node_colors = ['green' if n in [source, destination] else c for n, c in zip(G.nodes(), node_colors)]

nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.15, edge_color='gray')
nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=400, node_color=node_colors, edgecolors='black')
nx.draw_networkx_labels(G, pos, ax=ax1, font_size=9)

if path_dist:
    path_edges_d = list(zip(path_dist, path_dist[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=path_edges_d, 
                           edge_color='#ff4d4d', width=3, style='dashed', label='Distance Priority')

if path_batt:
    pos_shifted = {n: (x, y + 0.02) for n, (x, y) in pos.items()} 
    path_edges_b = list(zip(path_batt, path_batt[1:]))
    nx.draw_networkx_edges(G, pos_shifted, ax=ax1, edgelist=path_edges_b, 
                           edge_color='#3366ff', width=3, alpha=0.9, label='Battery Priority')

ax1.set_title("Central Danger Zone Scenario", fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.axis('off')

ax2 = plt.subplot(1, 2, 2)
categories = ['Network Lifetime', 'Avg Path Cost']
x = np.arange(len(categories))
width = 0.35

val_dist = [rounds_dist, avg_dist_dist]
val_batt = [rounds_batt, avg_dist_batt]

bar1 = ax2.bar(x - width/2, val_dist, width, label='Distance Priority', color='#ff4d4d', alpha=0.8)
bar2 = ax2.bar(x + width/2, val_batt, width, label='Battery Priority', color='#3366ff', alpha=0.8)

for rect in bar1 + bar2:
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., 1.01*height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

text_str = (
    f"Result:\n"
    f"• Red cuts through danger zone -> Dies fast.\n"
    f"• Blue detours safely -> Long life.\n"
    f"• Lifetime Winner: Battery Priority!"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax2.text(0.5, 0.85, text_str, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, ha='center')

ax2.set_title('Performance Analysis', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=11)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()