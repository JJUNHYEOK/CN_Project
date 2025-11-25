import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# 1. 환경 설정
random.seed(42)
np.random.seed(42)
num_nodes = 10 

# 2. 그래프 생성 및 데이터 설정
G = nx.complete_graph(num_nodes)
pos = nx.spring_layout(G, seed=42)

# 배터리 및 거리(Weight) 랜덤 설정
for node in G.nodes():
    G.nodes[node]['battery'] = random.randint(10, 100)

for u, v in G.edges():
    G.edges[u, v]['weight'] = round(random.uniform(1.0, 10.0), 1) # 거리 1.0~10.0

# --- [핵심] 시뮬레이션 함수 ---
def run_simulation(strategy):
    print(f"\n{'='*20} [{strategy.upper()} STRATEGY] {'='*20}")
    
    start_node = 0
    visited = [start_node]
    current = start_node
    total_cost = 0
    
    # 헤더 출력
    print(f"{'Step':<5} | {'Move':<10} | {'Cost':<6} | {'Target Batt':<12} | {'Total Cost'}")
    print("-" * 60)

    while len(visited) < num_nodes:
        candidates = [n for n in G.nodes() if n not in visited]
        
        next_node = None
        reason = ""
        
        if strategy == 'distance':
            # 거리가 가장 짧은 노드 선택
            next_node = min(candidates, key=lambda x: G[current][x]['weight'])
            reason = "(Closest)"
        elif strategy == 'battery':
            # 배터리가 가장 많은 노드 선택
            next_node = max(candidates, key=lambda x: G.nodes[x]['battery'])
            reason = "(Max Batt)"
            
        # 데이터 계산
        cost = G[current][next_node]['weight']
        target_battery = G.nodes[next_node]['battery']
        total_cost += cost
        
        # 터미널에 로그 출력
        print(f"{len(visited):<5} | {current} -> {next_node} | {cost:<6} | {target_battery}% {reason:<10} | {total_cost:.1f}")
        
        # 상태 업데이트
        visited.append(next_node)
        current = next_node
        
    print(f"-" * 60)
    print(f"✅ Final Result ({strategy}): Total Distance Cost = {total_cost:.1f}")
    return visited, total_cost

# 4. 실행
path_dist, cost_dist = run_simulation('distance')
path_batt, cost_batt = run_simulation('battery')

# --- [시각화] 그래프 그리기 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8)) # 그림 크기를 조금 더 키움
strategies = [('Distance Greedy', path_dist, cost_dist), 
              ('Battery Greedy', path_batt, cost_batt)]

for ax, (title, path, cost) in zip(axes, strategies):
    # 1. 배경 연결선 (수정됨: 더 선명하게)
    # alpha: 투명도 (0.1 -> 0.3으로 증가시켜 더 잘 보이게 함)
    # width: 선 두께 (0.5로 설정하여 얇지만 선명한 실선)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.0, edge_color='gray')
    
    # 2. 노드 그리기 (배터리 색상)
    batteries = [G.nodes[n]['battery'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, 
                           node_color=batteries, cmap=plt.cm.RdYlGn, 
                           vmin=0, vmax=100, edgecolors='black', linewidths=1.5)
    
    # 노드 안의 ID 숫자
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='black', font_weight='bold', font_size=10)

    # 3. 경로 그리기 (화살표)
    path_edges = list(zip(path, path[1:]))
    # Distance는 파랑, Battery는 보라
    main_color = 'blue' if 'Distance' in title else 'purple'
    
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges, 
                           edge_color=main_color, width=2.5, 
                           arrows=True, arrowstyle='-|>', arrowsize=25) # 화살표 크기 키움
    
    # 4. 순서 번호 표시 (노드 위쪽에 표시)
    for i, node in enumerate(path):
        x, y = pos[node]
        # 텍스트 위치를 노드 위로 조금 더 올림 (+0.1)
        ax.text(x, y+0.1, f"({i})", fontsize=12, color=main_color, fontweight='bold', ha='center')

    # 타이틀 및 테두리 제거
    ax.set_title(f"{title}\nTotal Cost: {cost:.1f}", fontsize=14, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()