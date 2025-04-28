#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <set>
#include<bits/stdc++.h>
using namespace std ; 
class Graph {
private:
    int n; // Number of vertices
    std::vector<std::vector<int>> adj; // Adjacency list
    
    // Helper function to check if a set of vertices forms a clique
    bool isClique(const std::vector<int>& vertices) const {
        for (size_t i = 0; i < vertices.size(); i++) {
            for (size_t j = i + 1; j < vertices.size(); j++) {
                if (std::find(adj[vertices[i]].begin(), adj[vertices[i]].end(), vertices[j]) == adj[vertices[i]].end()) {
                    return false;
                }
            }
        }
        return true;
    }
    
public:
    Graph(int vertices) : n(vertices) {
        adj.resize(n);
    }
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    int getVertexCount() const {
        return n;
    }
    
    const std::vector<std::vector<int>>& getAdjList() const {
        return adj;
    }
    
    // Find all h-cliques in the graph using backtracking
    void findCliques(int h, std::vector<int>& current, int start, std::vector<std::vector<int>>& cliques) const {
        if (current.size() == h) {
            if (isClique(current)) {
                cliques.push_back(current);
            }
            return;
        }
        
        for (int i = start; i < n; i++) {
            current.push_back(i);
            findCliques(h, current, i + 1, cliques);
            current.pop_back();
        }
    }
    
    // Calculate clique degree of a vertex (number of h-cliques containing v)
    int cliqueDegree(int v, int h, const std::vector<std::vector<int>>& hCliques) const {
        int degree = 0;
        for (const auto& clique : hCliques) {
            if (std::find(clique.begin(), clique.end(), v) != clique.end()) {
                degree++;
            }
        }
        return degree;
    }
    
    // Get induced subgraph from a set of vertices
    Graph getInducedSubgraph(const std::vector<int>& vertices) const {
        Graph subgraph(vertices.size());
        std::unordered_map<int, int> indexMap;
        
        for (size_t i = 0; i < vertices.size(); i++) {
            indexMap[vertices[i]] = i;
        }
        
        for (size_t i = 0; i < vertices.size(); i++) {
            for (size_t j = i + 1; j < vertices.size(); j++) {
                int u = vertices[i];
                int v = vertices[j];
                if (std::find(adj[u].begin(), adj[u].end(), v) != adj[u].end()) {
                    subgraph.addEdge(indexMap[u], indexMap[v]);
                }
            }
        }
        
        return subgraph;
    }
    
    // Calculate h-clique density
    double cliqueDensity(int h) const {
        std::vector<std::vector<int>> cliques;
        std::vector<int> temp;
        findCliques(h, temp, 0, cliques);
        
        if (n == 0) return 0.0;
        return static_cast<double>(cliques.size()) / n;
    }
    
    // Print the graph structure
    void printGraph() const {
        std::cout << "Graph structure:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "Vertex " << i << " connected to: ";
            for (int j : adj[i]) {
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Core decomposition algorithm for k-clique cores
// Core decomposition algorithm for k-clique cores - FIXED VERSION
std::vector<int> coreDecomposition(const Graph& G, int h) {
    int n = G.getVertexCount();
    std::vector<int> core(n, 0);

    // 1) Enumerate all h-cliques
    std::vector<std::vector<int>> hCliques;
    std::vector<int> tmp;
    G.findCliques(h, tmp, 0, hCliques);

    // 2) Build vertex→cliques map
    std::vector<std::vector<int>> vertexToCliques(n);
    for (int i = 0; i < (int)hCliques.size(); i++)
        for (int v : hCliques[i])
            vertexToCliques[v].push_back(i);

    // 3) Init degrees = number of cliques per vertex
    std::vector<int> degrees(n);
    for (int v = 0; v < n; v++)
        degrees[v] = vertexToCliques[v].size();

    // 4) Track which cliques are still active
    std::vector<int> cliqueSize(hCliques.size(), h);
    std::vector<bool> cliqueActive(hCliques.size(), true);

    // 5) Build a simple "queue" of (degree,vertex)
    std::vector<std::pair<int,int>> vertexQueue;
    vertexQueue.reserve(n);
    for (int v = 0; v < n; v++)
        vertexQueue.emplace_back(degrees[v], v);

    std::vector<bool> removed(n,false);

    // 6) Peel vertices in increasing order of current degree
    while (!vertexQueue.empty()) {
        // find and remove the min‐degree vertex
        auto it = std::min_element(vertexQueue.begin(), vertexQueue.end());
        int k = it->first;
        int v = it->second;
        vertexQueue.erase(it);

        removed[v] = true;
        core[v] = k;

        // for each clique containing v
        for (int ci : vertexToCliques[v]) {
            if (!cliqueActive[ci]) continue;

            // shrink clique; if it drops below h, invalidate it
            if (--cliqueSize[ci] < h) {
                cliqueActive[ci] = false;
                // only decrement neighbors with higher degree
                for (int u : hCliques[ci]) {
                    if (u == v || removed[u]) continue;
                    if (degrees[u] > k) {
                        degrees[u]--;
                        // update u’s entry in the queue
                        for (auto& vd : vertexQueue) {
                            if (vd.second == u) {
                                vd.first = degrees[u];
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return core;
}

// Extract k-core from a graph using core numbers
Graph extractKCore(const Graph& G, int k, const std::vector<int>& coreNumbers) {
    int n = G.getVertexCount();
    std::vector<int> kCoreVertices;
    
    for (int v = 0; v < n; v++) {
        if (coreNumbers[v] >= k) {
            kCoreVertices.push_back(v);
        }
    }
    
    std::cout << "Extracting " << k << "-core with " << kCoreVertices.size() << " vertices: ";
    for (int v : kCoreVertices) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    
    return G.getInducedSubgraph(kCoreVertices);
}

// Find connected components of a graph
std::vector<Graph> getConnectedComponents(const Graph& G) {
    int n = G.getVertexCount();
    std::vector<bool> visited(n, false);
    std::vector<Graph> components;
    
    for (int v = 0; v < n; v++) {
        if (!visited[v]) {
            std::vector<int> componentVertices;
            std::queue<int> q;
            q.push(v);
            visited[v] = true;
            
            while (!q.empty()) {
                int u = q.front();
                q.pop();
                componentVertices.push_back(u);
                
                for (int w : G.getAdjList()[u]) {
                    if (!visited[w]) {
                        visited[w] = true;
                        q.push(w);
                    }
                }
            }
            
            std::cout << "Component " << components.size() + 1 << " vertices: ";
            for (int u : componentVertices) {
                std::cout << u << " ";
            }
            std::cout << std::endl;
            
            components.push_back(G.getInducedSubgraph(componentVertices));
        }
    }
    
    return components;
}

// Build flow network for min-cut computation
std::vector<std::vector<int>> buildFlowNetwork(const Graph& G, int h, double alpha) {
    int n = G.getVertexCount();
    
    // Find all h-cliques and (h-1)-cliques
    std::vector<std::vector<int>> hCliques;
    std::vector<std::vector<int>> hMinus1Cliques;
    std::vector<int> temp;
    G.findCliques(h, temp, 0, hCliques);
    G.findCliques(h-1, temp, 0, hMinus1Cliques);
    
    std::cout << "Flow network: Found " << hCliques.size() << " " << h << "-cliques and " 
              << hMinus1Cliques.size() << " " << (h-1) << "-cliques" << std::endl;
    
    // Calculate clique degrees
    std::vector<int> cliqueDegrees(n, 0);
    for (int v = 0; v < n; v++) {
        cliqueDegrees[v] = G.cliqueDegree(v, h, hCliques);
    }
    
    // Build flow network
    int numNodes = 1 + n + hMinus1Cliques.size() + 1;
    std::vector<std::vector<int>> capacity(numNodes, std::vector<int>(numNodes, 0));
    
    int s = 0;
    int t = numNodes - 1;
    
    // Add edges from s to vertices
    for (int v = 0; v < n; v++) {
        capacity[s][v + 1] = cliqueDegrees[v];
        if (cliqueDegrees[v] > 0) {
            std::cout << "Edge s -> " << v << " with capacity " << cliqueDegrees[v] << std::endl;
        }
    }
    
    // Add edges from vertices to t
    for (int v = 0; v < n; v++) {
        capacity[v + 1][t] = alpha * h;
        std::cout << "Edge " << v << " -> t with capacity " << (alpha * h) << std::endl;
    }
    
    // Add edges from vertices to (h-1)-cliques and vice versa
    for (size_t i = 0; i < hMinus1Cliques.size(); i++) {
        const auto& clique = hMinus1Cliques[i];
        
        // Add edges from (h-1)-cliques to vertices
        for (int v : clique) {
            capacity[n + 1 + i][v + 1] = INT_MAX;
            std::cout << "Edge clique" << i << " -> " << v << " with capacity INF" << std::endl;
        }
        
        // Add edges from vertices to (h-1)-cliques
        for (int v = 0; v < n; v++) {
            if (std::find(clique.begin(), clique.end(), v) != clique.end()) {
                continue;
            }
            
            bool canFormClique = true;
            for (int u : clique) {
                if (std::find(G.getAdjList()[v].begin(), G.getAdjList()[v].end(), u) == G.getAdjList()[v].end()) {
                    canFormClique = false;
                    break;
                }
            }
            
            if (canFormClique) {
                capacity[v + 1][n + 1 + i] = 1;
                std::cout << "Edge " << v << " -> clique" << i << " with capacity 1" << std::endl;
            }
        }
    }
    
    return capacity;
}

// Ford-Fulkerson algorithm for max flow / min cut
int fordFulkerson(const std::vector<std::vector<int>>& capacity, int s, int t, std::vector<int>& minCut) {
    int n = capacity.size();
    std::vector<std::vector<int>> residual = capacity;
    std::vector<int> parent(n);
    int maxFlow = 0;
    
    // Fixed lambda function with explicit return type
    auto bfs = [&](std::vector<int>& parent) -> bool {
        std::vector<bool> visited(n, false);
        std::queue<int> q;
        
        q.push(s);
        visited[s] = true;
        parent[s] = -1;
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v = 0; v < n; v++) {
                if (!visited[v] && residual[u][v] > 0) {
                    q.push(v);
                    parent[v] = u;
                    visited[v] = true;
                    if (v == t) return true;
                }
            }
        }
        
        return bool(visited[t]); // Explicitly convert to bool
    };
    
    // Augment flow while there is a path from s to t
    while (bfs(parent)) {
        int pathFlow = INT_MAX;
        
        // Find minimum residual capacity along the path
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            pathFlow = std::min(pathFlow, residual[u][v]);
        }
        
        // Update residual capacities
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            residual[u][v] -= pathFlow;
            residual[v][u] += pathFlow;
        }
        
        maxFlow += pathFlow;
    }
    
    std::cout << "Max flow: " << maxFlow << std::endl;
    
    // Find min-cut
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    q.push(s);
    visited[s] = true;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v = 0; v < n; v++) {
            if (!visited[v] && residual[u][v] > 0) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    
    minCut.clear();
    for (int i = 0; i < n; i++) {
        if (visited[i]) {
            minCut.push_back(i);
        }
    }
    
    std::cout << "Min-cut vertices: ";
    for (int v : minCut) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    
    return maxFlow;
}

// CoreExact algorithm (Algorithm 4)
Graph coreExact(const Graph& G, int h) {
    int n = G.getVertexCount();
    std::cout << "Running CoreExact algorithm for " << h << "-clique densest subgraph" << std::endl;
    
    // Print graph structure for debugging
    G.printGraph();
    
    // Step 1: Perform core decomposition
    std::cout << "Performing core decomposition..." << std::endl;
    std::vector<int> coreNumbers = coreDecomposition(G, h);
    
    int kMax = 0;
    for (int k : coreNumbers) {
        kMax = std::max(kMax, k);
    }
    std::cout << "Maximum core number: " << kMax << std::endl;
    
    // Step 2: Find initial lower bound using (k',Ψ)-core
    double rho = 0.0;
    std::vector<std::vector<int>> hCliques;
    std::vector<int> temp;
    G.findCliques(h, temp, 0, hCliques);
    
    if (!hCliques.empty()) {
        rho = static_cast<double>(hCliques.size()) / n;
    }
    
    int kPrime = std::ceil(rho);
    std::cout << "Initial lower bound: " << rho << ", k': " << kPrime << std::endl;
    
    // Step 3: Extract (k',Ψ)-core
    Graph kPrimeCore = extractKCore(G, kPrime, coreNumbers);
    
    // Step 4: Get connected components
    std::vector<Graph> components = getConnectedComponents(kPrimeCore);
    std::cout << "Number of connected components: " << components.size() << std::endl;
    
    // Initialize best subgraph
    Graph bestSubgraph(0);
    double bestDensity = 0.0;
    
    // Step 5-20: Process each connected component
    for (size_t i = 0; i < components.size(); i++) {
        Graph component = components[i];
        std::cout << "Processing component " << i+1 << " with " << component.getVertexCount() << " vertices" << std::endl;
        
        // Skip tiny components
        if (component.getVertexCount() < h) {
            std::cout << "Component too small, skipping" << std::endl;
            continue;
        }
        
        // Step 7-8: Check if component has density >= rho
        double componentDensity = component.cliqueDensity(h);
        std::cout << "Component density: " << componentDensity << std::endl;
        
        if (componentDensity < rho) {
            std::cout << "Component density " << componentDensity << " < lower bound " << rho << ", skipping" << std::endl;
            continue;
        }
        
        // Step 10-19: Binary search for optimal density
        double l = 0;
        double u = kMax > 0 ? kMax : 1.0;  // Ensure u is positive
        std::vector<int> bestCut;
        
        std::cout << "Starting binary search with bounds [" << l << ", " << u << "]" << std::endl;
        
        while (u - l >= 1.0 / (component.getVertexCount() * (component.getVertexCount() - 1))) {
            double alpha = (l + u) / 2.0;
            std::cout << "Trying α = " << alpha << std::endl;
            
            // Build flow network and find min-cut
            std::vector<std::vector<int>> flowNetwork = buildFlowNetwork(component, h, alpha);
            std::vector<int> minCut;
            fordFulkerson(flowNetwork, 0, flowNetwork.size()-1, minCut);
            
            if (minCut.size() <= 1) {
                // Only source is in the cut
                u = alpha;
                std::cout << "Cut contains only source, reducing upper bound to " << u << std::endl;
            } else {
                // Extract vertices from the cut (excluding source)
                std::vector<int> cutVertices;
                for (int node : minCut) {
                    if (node != 0 && node < component.getVertexCount() + 1) {
                        cutVertices.push_back(node - 1);
                    }
                }
                
                l = alpha;
                bestCut = cutVertices;
                std::cout << "Cut contains " << cutVertices.size() << " vertices, increasing lower bound to " << l << std::endl;
            }
        }
        
        // Extract subgraph from best cut
        if (!bestCut.empty()) {
            Graph candidateSubgraph = component.getInducedSubgraph(bestCut);
            double candidateDensity = candidateSubgraph.cliqueDensity(h);
            
            std::cout << "Candidate subgraph has " << candidateSubgraph.getVertexCount() 
                      << " vertices and density " << candidateDensity << std::endl;
            
            if (candidateDensity > bestDensity) {
                bestDensity = candidateDensity;
                bestSubgraph = candidateSubgraph;
                std::cout << "Found better subgraph with density " << bestDensity << std::endl;
            }
        }
    }
    
    std::cout << "CoreExact completed. Best subgraph has " << bestSubgraph.getVertexCount() 
              << " vertices and density " << bestDensity << std::endl;
    
    return bestSubgraph;
}

int main() {
    std::ifstream fin("CA-HepTh.txt");
    if (!fin) {
        std::cerr << "Error: cannot open input.txt\n";
        return 1;
    }

    // 2) Read n, m, h
    int n, m, h;
    fin >> n >> m >> h;
    std::cout << "Read: n=" << n << "  m=" << m << "  h=" << h << "\n";

    // 3) Prepare mapping from external IDs to internal [0..n-1]
    std::unordered_map<int,int> ext2int;
    ext2int.reserve(n);
    std::vector<int> int2ext;
    int2ext.reserve(n);
    
    // 4) Create graph with n internal vertices
    Graph G(n);

    // 5) Read edges, map external→internal, add to G
    for (int i = 0; i < m; i++) {
        int ue, ve;
        fin >> ue >> ve;

        // map ue
        auto it = ext2int.find(ue);
        int ui;
        if (it == ext2int.end()) {
            ui = ext2int[ue] = (int)int2ext.size();
            int2ext.push_back(ue);
        } else {
            ui = it->second;
        }

        // map ve
        it = ext2int.find(ve);
        int vi;
        if (it == ext2int.end()) {
            vi = ext2int[ve] = (int)int2ext.size();
            int2ext.push_back(ve);
        } else {
            vi = it->second;
        }

        // add edge in internal graph
        G.addEdge(ui, vi);
    }
    fin.close();
    
    // Find the h-clique densest subgraph
    auto startTime = std::chrono::high_resolution_clock::now();
    Graph densestSubgraph = coreExact(G, h);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    double executionTime = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "Execution time: " << executionTime << " seconds" << std::endl;
    
    return 0;
}