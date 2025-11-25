import sys
import struct
import time

# --- 导入修复 ---
try:
    # 尝试较新版本的路径
    from ortools.linear_solver.python import model_builder
except ImportError:
    try:
        # 尝试旧版本或开发版路径
        from ortools.model_builder.python import model_builder
    except ImportError:
        print("Error: Could not import 'model_builder'. Please update OR-Tools.")
        sys.exit(1)
# ------------------

def read_graph_file(filename):
    edges = []
    nodes_seen = set()
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    u, v = int(parts[0]), int(parts[1])
                    cap = float(parts[2])
                    edges.append({'src': u, 'dst': v, 'capacity': cap})
                    nodes_seen.add(u)
                    nodes_seen.add(v)
    except FileNotFoundError:
        print(f"Error: Cannot open graph file: {filename}")
        sys.exit(1)

    if not edges:
        return [], 0, False

    min_node = min(nodes_seen)
    max_node = max(nodes_seen)
    has_zero = 0 in nodes_seen
    is_one_indexed = False
    
    if min_node == 1 and not has_zero:
        is_one_indexed = True
        print("Graph appears 1-indexed. Converting to 0-indexed.")
        for edge in edges:
            edge['src'] -= 1
            edge['dst'] -= 1
        num_nodes = max_node
    else:
        num_nodes = max_node + 1

    print(f"Read {len(edges)} edges with {num_nodes} nodes.")
    return edges, num_nodes, is_one_indexed

def read_demand_file(filename, is_one_indexed):
    commodities = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    s, t = int(parts[0]), int(parts[1])
                    d = float(parts[2])
                    
                    if is_one_indexed:
                        s -= 1
                        t -= 1
                    
                    commodities.append({'src': s, 'dst': t, 'demand': d})
    except FileNotFoundError:
        print(f"Error: Cannot open demand file: {filename}")
        sys.exit(1)

    print(f"Read {len(commodities)} commodities.")
    return commodities

def solve_mcf_model_builder(nodes_count, edges, commodities):
    print(f"Initializing ModelBuilder...")
    model = model_builder.ModelBuilder() 
    
    num_edges = len(edges)
    num_commodities = len(commodities)
    num_vars = num_edges * num_commodities

    print(f"Creating {num_vars} variables...")
    
    # --- 修复部分开始 ---
    # 错误原因: ModelBuilder 没有 new_num_var_array 方法。
    # 修复: 使用列表推导式循环创建变量。
    # 注意：对于 1亿+ 变量，这一步在 Python 中可能需要几分钟，但在没有 pandas 的情况下是标准做法。
    # 变量范围 [0, 无穷大]，容量约束单独添加。
    all_flow_vars = [model.new_num_var(0.0, 1e18, "") for _ in range(num_vars)]
    # --- 修复部分结束 ---
    
    print("Building capacity constraints...")
    # 1. 容量约束: sum(flow[e, k]) <= capacity[e]
    for i in range(num_edges):
        start_idx = i * num_commodities
        end_idx = start_idx + num_commodities
        # 切片获取该边的所有商品流变量
        edge_vars = all_flow_vars[start_idx:end_idx]
        
        # 使用 LinearExpr.sum 求和
        model.add(model_builder.LinearExpr.sum(edge_vars) <= edges[i]['capacity'])

    print("Building flow conservation constraints...")
    # 预处理邻接表
    out_edges = [[] for _ in range(nodes_count)]
    in_edges = [[] for _ in range(nodes_count)]
    for i, edge in enumerate(edges):
        out_edges[edge['src']].append(i)
        in_edges[edge['dst']].append(i)

    # 2. 流量守恒约束
    for k in range(num_commodities):
        s_node = commodities[k]['src']
        t_node = commodities[k]['dst']
        demand_val = commodities[k]['demand']
        
        for u in range(nodes_count):
            net_required = 0.0
            if u == s_node:
                net_required = demand_val
            elif u == t_node:
                net_required = -demand_val
            
            # 优化：跳过无需求且无连接的节点
            if net_required == 0 and not out_edges[u] and not in_edges[u]:
                continue

            # 收集相关变量
            term_out = [all_flow_vars[e_idx * num_commodities + k] for e_idx in out_edges[u]]
            term_in = [all_flow_vars[e_idx * num_commodities + k] for e_idx in in_edges[u]]
            
            # 约束: 流出 - 流入 = 需求
            model.add(model_builder.LinearExpr.sum(term_out) - model_builder.LinearExpr.sum(term_in) == net_required)

    print(f"Model built.")
    
    print("Configuring PDLP solver for feasibility...")
    solver = model_builder.ModelSolver('PDLP')
    
    if not solver.solver_is_supported():
        print("PDLP solver is not supported in this environment.")
        return None

    solver.enable_output(True)
    # 设置求解时间限制（例如 1 小时），防止无限运行
    solver.set_time_limit_in_seconds(7200) 
    

    # 设置 PDLP 求解精度为 1e-6
    solver.set_solver_specific_parameters("termination_criteria { eps_optimal_absolute: 1.0e-6 eps_optimal_relative: 1.0e-6 }")



    print("Starting solve...")
    start_time = time.time()
    status = solver.solve(model)
    end_time = time.time()
    print(f"\nTotal solver time: {end_time - start_time:.4f} seconds")

    if status == model_builder.SolveStatus.OPTIMAL or status == model_builder.SolveStatus.FEASIBLE:
        print(f"Solution found! Status: {status}")
        
        final_flows = {}
        print("Extracting solution values...")
        
        # 提取非零流量
        for i in range(num_edges):
            for k in range(num_commodities):
                idx = i * num_commodities + k
                val = solver.value(all_flow_vars[idx])
                if val > 1e-6:
                    final_flows[(i, k)] = val
        return final_flows
    else:
        print(f"Solution not found. Status: {status}")
        return None

def write_binary_output(filename, flows, edges):
    fmt = '<iiif' 
    try:
        with open(filename, 'wb') as f:
            print(f"\nWriting non-zero flows to binary file: {filename}")
            count = 0
            for (edge_idx, k), flow_val in flows.items():
                src = edges[edge_idx]['src']
                dst = edges[edge_idx]['dst']
                data = struct.pack(fmt, src, dst, k, flow_val)
                f.write(data)
                count += 1
            print(f"Solution saved. ({count} records)")
    except IOError as e:
        print(f"Error: Could not open output file {filename}: {e}")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <graph_file> <demand_file> [output_file]")
        sys.exit(1)

    graph_file = sys.argv[1]
    demand_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Loading graph from {graph_file}...")
    edges, num_nodes, is_one_indexed = read_graph_file(graph_file)
    
    print(f"Loading demands from {demand_file}...")
    commodities = read_demand_file(demand_file, is_one_indexed)

    result_flows = solve_mcf_model_builder(num_nodes, edges, commodities)

    if result_flows is not None and output_file:
        write_binary_output(output_file, result_flows, edges)

if __name__ == "__main__":
    main()