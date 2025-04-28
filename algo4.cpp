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
    int n; 
    vector<vector<int>> adj;
    
    bool isC(const vector<int>& vertices) const {
        for (size_t i = 0; i < vertices.size(); i++) {
            for (size_t j = i + 1; j < vertices.size(); j++) {
                if (find(adj[vertices[i]].begin(), adj[vertices[i]].end(), vertices[j]) == adj[vertices[i]].end()) {
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
    
    const vector<vector<int>>& getAdjList() const {
        return adj;
    }
    
    void fc(int h, vector<int>& current, int start, vector<vector<int>>& cliques) const {
        if (current.size() == h) {
            if (isC(current)) {
                cliques.push_back(current);
            }
            return;
        }
        
        for (int i = start; i < n; i++) {
            current.push_back(i);
            fc(h, current, i + 1, cliques);
            current.pop_back();
        }
    }
    
    int CD(int v, int h, const vector<vector<int>>& hCliques) const {
        int degree = 0;
        for (const auto& clique : hCliques) {
            if (find(clique.begin(), clique.end(), v) != clique.end()) {
                degree++;
            }
        }
        return degree;
    }
    
    Graph GIS(const vector<int>& vertices) const {
        Graph subgraph(vertices.size());
        unordered_map<int, int> indexMap;
        
        for (size_t i = 0; i < vertices.size(); i++) {
            indexMap[vertices[i]] = i;
        }
        
        for (size_t i = 0; i < vertices.size(); i++) {
            for (size_t j = i + 1; j < vertices.size(); j++) {
                int u = vertices[i];
                int v = vertices[j];
                if (find(adj[u].begin(), adj[u].end(), v) != adj[u].end()) {
                    subgraph.addEdge(indexMap[u], indexMap[v]);
                }
            }
        }
        
        return subgraph;
    }
    
    double cliqueDensity(int h) const {
        vector<vector<int>> cliques;
        vector<int> temp;
        fc(h, temp, 0, cliques);
        
        if (n == 0) return 0.0;
        return static_cast<double>(cliques.size()) / n;
    }
    
    void printGraph() const {
        cout << "Graph structure:" << endl;
        for (int i = 0; i < n; i++) {
            cout << "Vertex " << i << " connected to: ";
            for (int j : adj[i]) {
                cout << j << " ";
            }
            cout << endl;
        }
    }
};

vector<int> coreDecomposition(const Graph& G, int h) {
    int n = G.getVertexCount();
    vector<int> core(n, 0);

    vector<vector<int>> hCliques;
    vector<int> tmp;
    G.fc(h, tmp, 0, hCliques);

    vector<vector<int>> vertexToCliques(n);
    for (int i = 0; i < (int)hCliques.size(); i++)
        for (int v : hCliques[i])
            vertexToCliques[v].push_back(i);

    vector<int> degrees(n);
    for (int v = 0; v < n; v++)
        degrees[v] = vertexToCliques[v].size();

    vector<int> cliqueSize(hCliques.size(), h);
    vector<bool> cliqueActive(hCliques.size(), true);

    vector<pair<int,int>> vertexQueue;
    vertexQueue.reserve(n);
    for (int v = 0; v < n; v++)
        vertexQueue.emplace_back(degrees[v], v);

    vector<bool> removed(n,false);

    while (!vertexQueue.empty()) {
        auto it = min_element(vertexQueue.begin(), vertexQueue.end());
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

Graph extractKCore(const Graph& G, int k, const vector<int>& coreNumbers) {
    int n = G.getVertexCount();
    vector<int> kCoreVertices;
    
    for (int v = 0; v < n; v++) {
        if (coreNumbers[v] >= k) {
            kCoreVertices.push_back(v);
        }
    }
    
    cout << "Extracting " << k << "-core with " << kCoreVertices.size() << " vertices: ";
    for (int v : kCoreVertices) {
        cout << v << " ";
    }
    cout << endl;
    
    return G.GIS(kCoreVertices);
}

// Find connected components of a graph
vector<Graph> getConnectedComponents(const Graph& G) {
    int n = G.getVertexCount();
    vector<bool> visited(n, false);
    vector<Graph> components;
    
    for (int v = 0; v < n; v++) {
        if (!visited[v]) {
            vector<int> componentVertices;
            queue<int> q;
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
            
            cout << "Component " << components.size() + 1 << " vertices: ";
            for (int u : componentVertices) {
                cout << u << " ";
            }
            cout << endl;
            
            components.push_back(G.GIS(componentVertices));
        }
    }
    
    return components;
}

vector<vector<int>> buildFlowNetwork(const Graph& G, int h, double alpha) {
    int n = G.getVertexCount();
    
    vector<vector<int>> hCliques;
    vector<vector<int>> hMinus1Cliques;
    vector<int> temp;
    G.fc(h, temp, 0, hCliques);
    G.fc(h-1, temp, 0, hMinus1Cliques);
    
    cout << "Flow network: Found " << hCliques.size() << " " << h << "-cliques and " 
              << hMinus1Cliques.size() << " " << (h-1) << "-cliques" << endl;
    
    vector<int> CDs(n, 0);
    for (int v = 0; v < n; v++) {
        CDs[v] = G.CD(v, h, hCliques);
    }
    
    int numNodes = 1 + n + hMinus1Cliques.size() + 1;
    vector<vector<int>> capacity(numNodes, vector<int>(numNodes, 0));
    
    int s = 0;
    int t = numNodes - 1;
    
    for (int v = 0; v < n; v++) {
        capacity[s][v + 1] = CDs[v];
        if (CDs[v] > 0) {
            cout << "Edge s -> " << v << " with capacity " << CDs[v] << endl;
        }
    }
    
    for (int v = 0; v < n; v++) {
        capacity[v + 1][t] = alpha * h;
        cout << "Edge " << v << " -> t with capacity " << (alpha * h) << endl;
    }
    
    for (size_t i = 0; i < hMinus1Cliques.size(); i++) {
        const auto& clique = hMinus1Cliques[i];
        
        for (int v : clique) {
            capacity[n + 1 + i][v + 1] = INT_MAX;
            cout << "Edge clique" << i << " -> " << v << " with capacity INF" << endl;
        }
        
        for (int v = 0; v < n; v++) {
            if (find(clique.begin(), clique.end(), v) != clique.end()) {
                continue;
            }
            
            bool canFormClique = true;
            for (int u : clique) {
                if (find(G.getAdjList()[v].begin(), G.getAdjList()[v].end(), u) == G.getAdjList()[v].end()) {
                    canFormClique = false;
                    break;
                }
            }
            
            if (canFormClique) {
                capacity[v + 1][n + 1 + i] = 1;
                cout << "Edge " << v << " -> clique" << i << " with capacity 1" << endl;
            }
        }
    }
    
    return capacity;
}

int ff(const vector<vector<int>>& capacity, int s, int t, vector<int>& minCut) {
    int n = capacity.size();
    vector<vector<int>> residual = capacity;
    vector<int> parent(n);
    int maxFlow = 0;
    
    auto bfs = [&](vector<int>& parent) -> bool {
        vector<bool> visited(n, false);
        queue<int> q;
        
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
        
        return bool(visited[t]); 
    };
    
    while (bfs(parent)) {
        int pathFlow = INT_MAX;
        
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            pathFlow = min(pathFlow, residual[u][v]);
        }
        
        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            residual[u][v] -= pathFlow;
            residual[v][u] += pathFlow;
        }
        
        maxFlow += pathFlow;
    }
    
    cout << "Max flow: " << maxFlow << endl;
    
    // Find min-cut
    vector<bool> visited(n, false);
    queue<int> q;
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
    
    cout << "Min-cut vertices: ";
    for (int v : minCut) {
        cout << v << " ";
    }
    cout << endl;
    
    return maxFlow;
}

// (Algorithm 4)
Graph coreExact(const Graph& G, int h) {
    int n = G.getVertexCount();
    cout << "Running CoreExact algorithm for " << h << "-clique densest subgraph" << endl;
    
    G.printGraph();
    
    cout << "Performing core decomposition..." << endl;
    vector<int> coreNumbers = coreDecomposition(G, h);
    
    int kMax = 0;
    for (int k : coreNumbers) {
        kMax = max(kMax, k);
    }
    cout << "Maximum core number: " << kMax << endl;
    
    double rho = 0.0;
    vector<vector<int>> hCliques;
    vector<int> temp;
    G.fc(h, temp, 0, hCliques);
    
    if (!hCliques.empty()) {
        rho = static_cast<double>(hCliques.size()) / n;
    }
    
    int kPrime = ceil(rho);
    cout << "Initial lower bound: " << rho << ", k': " << kPrime << endl;
    
    Graph kPrimeCore = extractKCore(G, kPrime, coreNumbers);
    
    vector<Graph> components = getConnectedComponents(kPrimeCore);
    cout << "Number of connected components: " << components.size() << endl;
    
    Graph bestSubgraph(0);
    double bestDensity = 0.0;
    
    for (size_t i = 0; i < components.size(); i++) {
        Graph component = components[i];
        cout << "Processing component " << i+1 << " with " << component.getVertexCount() << " vertices" << endl;
        
        // Skip tiny components
        if (component.getVertexCount() < h) {
            cout << "Component too small, skipping" << endl;
            continue;
        }
        
        double componentDensity = component.cliqueDensity(h);
        cout << "Component density: " << componentDensity << endl;
        
        if (componentDensity < rho) {
            cout << "Component density " << componentDensity << " < lower bound " << rho << ", skipping" << endl;
            continue;
        }
        
        double l = 0;
        double u = kMax > 0 ? kMax : 1.0;  
        vector<int> bestCut;
        
        cout << "Starting binary search with bounds [" << l << ", " << u << "]" << endl;
        
        while (u - l >= 1.0 / (component.getVertexCount() * (component.getVertexCount() - 1))) {
            double alpha = (l + u) / 2.0;
            cout << "Trying α = " << alpha << endl;
            
            vector<vector<int>> flowNetwork = buildFlowNetwork(component, h, alpha);
            vector<int> minCut;
            ff
        (flowNetwork, 0, flowNetwork.size()-1, minCut);
            
            if (minCut.size() <= 1) {
                u = alpha;
                cout << "Cut contains only source, reducing upper bound to " << u << endl;
            } else {
                vector<int> cutVertices;
                for (int node : minCut) {
                    if (node != 0 && node < component.getVertexCount() + 1) {
                        cutVertices.push_back(node - 1);
                    }
                }
                
                l = alpha;
                bestCut = cutVertices;
                cout << "Cut contains " << cutVertices.size() << " vertices, increasing lower bound to " << l << endl;
            }
        }
        
        // Extract subgraph from best cut
        if (!bestCut.empty()) {
            Graph candidateSubgraph = component.GIS(bestCut);
            double candidateDensity = candidateSubgraph.cliqueDensity(h);
            
            cout << "Candidate subgraph has " << candidateSubgraph.getVertexCount() 
                      << " vertices and density " << candidateDensity << endl;
            
            if (candidateDensity > bestDensity) {
                bestDensity = candidateDensity;
                bestSubgraph = candidateSubgraph;
                cout << "Found better subgraph with density " << bestDensity << endl;
            }
        }
    }
    
    cout << "CoreExact completed. Best subgraph has " << bestSubgraph.getVertexCount() 
              << " vertices and density " << bestDensity << endl;
    
    return bestSubgraph;
}

int main() {
    ifstream fin("net.txt");
    if (!fin) {
        cerr << "Error: cannot open input.txt\n";
        return 1;
    }

    int n, m, h;
    fin >> n >> m >> h;
    cout << "Read: n=" << n << "  m=" << m << "  h=" << h << "\n";

    unordered_map<int,int> exe11;
    exe11.reserve(n);
    vector<int> int2ext;
    int2ext.reserve(n);
    
    Graph G(n);

    for (int i = 0; i < m; i++) {
        int ue, ve;
        fin >> ue >> ve;

        
        auto it = exe11.find(ue);
        int ui;
        if (it == exe11.end()) {
            ui = exe11[ue] = (int)int2ext.size();
            int2ext.push_back(ue);
        } else {
            ui = it->second;
        }

        
        it = exe11.find(ve);
        int vi;
        if (it == exe11.end()) {
            vi = exe11[ve] = (int)int2ext.size();
            int2ext.push_back(ve);
        } else {
            vi = it->second;
        }

    
        G.addEdge(ui, vi);
    }
    fin.close();
    
    auto startTime = chrono::high_resolution_clock::now();
    Graph densestSubgraph = coreExact(G, h);
    auto endTime = chrono::high_resolution_clock::now();
    
    double executionTime = chrono::duration<double>(endTime - startTime).count();
    cout << "Execution time: " << executionTime << " seconds" << endl;
    
    return 0;
}
