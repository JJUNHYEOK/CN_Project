import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# 1. 환경 설정
random.seed(42)
np.random.seed(42)
num_nodes = 15 
plt.style.use('default') 

# 2. 그래프 생성
G = nx.random_geometric_graph(num_nodes, radius=0.5, seed=42)
pos = nx.get_node_attributes(G, 'pos')

# 거리(Weight) 계산 및 초기 배터리 설정
for u, v in G.edges():
    dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) * 10 # 거리 스케일링
    G.edges[u, v]['weight'] = round(dist, 1)

# 초기 배터리 설정 (모두 100으로 시작)
INITIAL_BATTERY = 100
for node in G.nodes():
    G.nodes[node]['battery'] = INITIAL_BATTERY


def run_lifetime_simulation(graph, strategy):
  
    G_sim = graph.copy()
    source = 0
    destination = max(G_sim.nodes(), key=lambda n: np.linalg.norm(np.array(pos[n]) - np.array(pos[source])))
    
    rounds = 0
    total_distance = 0
    path_history = []
    
    print(f"\nSimulation Start: {strategy} (Source {source} -> Dest {destination})")
    
    # 네트워크 생존 루프: 모든 노드의 배터리가 0보다 클 때까지 반복
    while True:
        current = source
        path = [current]
        round_cost = 0
        
        # 패킷 1회 전송 (Source -> Dest)
        while current != destination:
            neighbors = list(G_sim.neighbors(current))
            # 이미 방문한 노드 제외 (루프 방지)
            valid_neighbors = [n for n in neighbors if n not in path]
            
            # 갈 곳이 없거나 배터리가 없으면 패킷 전송 실패 (해당 라운드 종료)
            if not valid_neighbors:
                break

            next_node = None
            
            if strategy == 'Distance Priority':
                # 목적지까지의 거리가 가장 가까워지는 이웃 선택 (Greedy Forwarding)
                # (단순 거리 가중치 합이 아니라, 목적지와의 직선 거리가 짧은 순)
                next_node = min(valid_neighbors, key=lambda n: np.linalg.norm(np.array(pos[n]) - np.array(pos[destination])))
                
            elif strategy == 'Battery Priority':
                # 1순위: 배터리 많은 순, 2순위: 거리가 가까운 순
                # 배터리 가중치를 크게 둠
                next_node = max(valid_neighbors, key=lambda n: (G_sim.nodes[n]['battery'], -G_sim.edges[current, n]['weight']))
            
            # 이동 비용 계산 및 배터리 차감
            dist = G_sim.edges[current, next_node]['weight']
            energy_consump = dist * 1.5 # 거리 * 1.5 만큼 배터리 소모
            
            if G_sim.nodes[next_node]['battery'] <= energy_consump:
                # 배터리 부족으로 노드 사망 -> 시뮬레이션 종료 신호
                print(f"Node {next_node} Depleted! Simulation End.")
                return rounds, total_distance, path, G_sim
            
            G_sim.nodes[next_node]['battery'] -= energy_consump
            round_cost += dist
            current = next_node
            path.append(current)
        
        # 목적지 도착 성공 시 통계 업데이트
        if current == destination:
            rounds += 1
            total_distance += round_cost
            path_history = path # 마지막 성공 경로 저장
        else:
            # 경로 단절로 인한 종료
            print("Path Broken.")
            break
            
    return rounds, total_distance, path_history, G_sim

# 3. 실행 (거리 우선 vs 배터리 우선)
rounds_dist, dist_dist, path_dist, G_final_dist = run_lifetime_simulation(G, 'Distance Priority')
rounds_batt, dist_batt, path_batt, G_final_batt = run_lifetime_simulation(G, 'Battery Priority')

# 평균 거리 계산
avg_dist_dist = dist_dist / rounds_dist if rounds_dist > 0 else 0
avg_dist_batt = dist_batt / rounds_batt if rounds_batt > 0 else 0

fig = plt.figure(figsize=(16, 8))
plt.subplots_adjust(wspace=0.3)

# 1. 왼쪽: 경로 토폴로지 비교 (마지막 성공 경로)
ax1 = plt.subplot(1, 2, 1)
nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2, edge_color='gray')
nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=300, node_color='lightgray')
nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)

# Distance 경로 (빨강 점선)
if path_dist:
    path_edges_d = list(zip(path_dist, path_dist[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=path_edges_d, edge_color='red', width=2, style='dashed', label='Distance Path')
    
# Battery 경로 (파랑 실선)
if path_batt:
    path_edges_b = list(zip(path_batt, path_batt[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=path_edges_b, edge_color='blue', width=2.5, alpha=0.8, label='Battery Path')

ax1.set_title("Route Comparison: Shortest(Red) vs Battery(Blue)", fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.axis('off')

# 2. 오른쪽: 성능 지표 막대 그래프 (이중 Y축)
ax2 = plt.subplot(1, 2, 2)
categories = ['Network Lifetime\n(Total Rounds)', 'Path Efficiency\n(Avg Distance)']
x = np.arange(len(categories))
width = 0.35

val_dist = [rounds_dist, avg_dist_dist]
val_batt = [rounds_batt, avg_dist_batt]

bar1 = ax2.bar(x - width/2, val_dist, width, label='Distance Priority', color='red', alpha=0.7)
bar2 = ax2.bar(x + width/2, val_batt, width, label='Battery Priority', color='blue', alpha=0.7)

ax2.set_ylabel('Value', fontsize=12)
ax2.set_title('Performance Trade-off Analysis', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=11)
ax2.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(bar1)
autolabel(bar2)

text_str = (
    f"Analysis:\n"
    f"1. Battery Algo survived {rounds_batt - rounds_dist} more rounds.\n"
    f"2. But Avg Distance increased by {avg_dist_batt - avg_dist_dist:.1f}.\n"
    f"-> Improved Lifetime by sacrificing Efficiency."
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.5, 0.85, text_str, transform=ax2.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, ha='center')

plt.tight_layout()
plt.show()