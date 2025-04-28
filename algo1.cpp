#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace std;

// Class to represent a graph
class Graph
{
private:
    int n;                   // Number of vertices
    vector<vector<int>> adj; 

    mutable vector<vector<int>> hCliquesCache;
    mutable vector<vector<int>> vertexToCliqueMap;
    mutable bool isCacheInitialized = false;

    bool isClique(const vector<int> &vertices) const
    {
        for (size_t i = 0; i < vertices.size(); i++)
        {
            for (size_t j = i + 1; j < vertices.size(); j++)
            {
                if (find(adj[vertices[i]].begin(), adj[vertices[i]].end(), vertices[j]) == adj[vertices[i]].end())
                {
                    return false;
                }
            }
        }
        return true;
    }

public:
    Graph(int vertices) : n(vertices)
    {
        adj.resize(n);
    }

    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int retrieveVertexCount() const
    {
        return n;
    }

    const vector<vector<int>> &getAdjList() const
    {
        return adj;
    }

    // Find all h-cliques
    void findCliques(int h, vector<int> &current, int start, vector<vector<int>> &cliques) const
    {
        static int counter = 0;
        counter++;

        // progresss reporting
        if (counter % 10000 == 0)
        {
            cout << "." << flush;
            if (counter % 500000 == 0)
            {
                cout << " [" << counter << " iterations]\n";
            }
        }

        if (current.size() == h)
        {
            if (isClique(current))
            {
                cliques.push_back(current);
            }
            return;
        }

        for (int i = start; i < n; i++)
        {
            current.push_back(i);
            findCliques(h, current, i + 1, cliques);
            current.pop_back();
        }
    }

    void initializeCliqueCache(int h) const
    {
        if (isCacheInitialized)
            return;

        cout << "Precomputing " << h << "-cliques..." << flush;
        auto start = chrono::high_resolution_clock::now();

        hCliquesCache.clear();
        vertexToCliqueMap.resize(n);
        vector<int> temp;
        findCliques(h, temp, 0, hCliquesCache);

        // Build mapping from vertices to cliques they belong to
        for (size_t i = 0; i < hCliquesCache.size(); i++)
        {
            for (int v : hCliquesCache[i])
            {
                vertexToCliqueMap[v].push_back(i);
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        cout << " Done! Found " << hCliquesCache.size() << " cliques in " << duration << "ms\n";
        isCacheInitialized = true;
    }

    int cliqueDegree(int v, int h) const
    {
        initializeCliqueCache(h);
        return vertexToCliqueMap[v].size();
    }

    int findMaxCliqueDegree(int h) const
    {
        initializeCliqueCache(h);

        int maximumCliqueDegree = 0;
        for (int v = 0; v < n; v++)
        {
            maximumCliqueDegree = max(maximumCliqueDegree, static_cast<int>(vertexToCliqueMap[v].size()));
        }
        return maximumCliqueDegree;
    }

    // Count h-cliques in the graph
    int countCliques(int h) const
    {
        initializeCliqueCache(h);
        return hCliquesCache.size();
    }

    // Calculate h-clique density
    double cliqueDesity(int h) const
    {
        int cliqueCount = countCliques(h);
        if (n == 0)
            return 0.0;
        return static_cast<double>(cliqueCount) / n;
    }

    // Print the graph
    void printGraph() const
    {
        for (int i = 0; i < n; i++)
        {
            cout << "Vertex " << i << ": ";
            for (int v : adj[i])
            {
                cout << v << " ";
            }
            cout << endl;
        }
    }

    Graph getInducedSubgraph(const vector<int> &vertices) const
    {
        Graph subgraph(vertices.size());
        unordered_map<int, int> indexMap;

        for (size_t i = 0; i < vertices.size(); i++)
        {
            indexMap[vertices[i]] = i;
        }

        for (size_t i = 0; i < vertices.size(); i++)
        {
            for (size_t j = i + 1; j < vertices.size(); j++)
            {
                int u = vertices[i];
                int v = vertices[j];
                if (find(adj[u].begin(), adj[u].end(), v) != adj[u].end())
                {
                    subgraph.addEdge(indexMap[u], indexMap[v]);
                }
            }
        }

        return subgraph;
    }
};

// Ford-Fulkerson algorithm to find max flow
int fordFulkerson(const vector<vector<int>> &capacity, int s, int t, vector<int> &minCut)
{
    int n = capacity.size();
    vector<vector<int>> residual(n, vector<int>(n, 0));

    // Initialize residual capacity
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            residual[i][j] = capacity[i][j];
        }
    }

    vector<int> parent(n);
    int maxFlow = 0;

    // BFS to find augmenting path
    auto bfs = [&](vector<int> &parent)
    {
        vector<bool> visited(n, false);
        queue<int> q;

        q.push(s);
        visited[s] = true;
        parent[s] = -1;

        while (!q.empty())
        {
            int u = q.front();
            q.pop();

            for (int v = 0; v < n; v++)
            {
                if (!visited[v] && residual[u][v] > 0)
                {
                    q.push(v);
                    parent[v] = u;
                    visited[v] = true;
                }
            }
        }

        return visited[t];
    };

    int fordFulkersonIterations = 0;
    cout << "Running max-flow algorithm: " << flush;

    while (bfs(parent))
    {
        fordFulkersonIterations++;
        if (fordFulkersonIterations % 100 == 0)
        {
            cout << "." << flush;
        }

        int pathFlow = numeric_limits<int>::max();

        for (int v = t; v != s; v = parent[v])
        {
            int u = parent[v];
            pathFlow = min(pathFlow, residual[u][v]);
        }
        // Update residual capacities
        for (int v = t; v != s; v = parent[v])
        {
            int u = parent[v];
            residual[u][v] -= pathFlow;
            residual[v][u] += pathFlow;
        }

        maxFlow += pathFlow;
    }

    cout << " Done!\n";

    // Find min-cut
    vector<bool> visited(n, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;

    while (!q.empty())
    {
        int u = q.front();
        q.pop();

        for (int v = 0; v < n; v++)
        {
            if (residual[u][v] > 0 && !visited[v])
            {
                visited[v] = true;
                q.push(v);
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        if (visited[i])
        {
            minCut.push_back(i);
        }
    }

    return maxFlow;
}

// Find the Clique Densest Subgraph with progress indicators
Graph locateCliqueDensitySubgraph(const Graph &G, int h)
{
    int n = G.retrieveVertexCount();
    cout << "Analyzing graph with " << n << " vertices for " << h << "-clique densest subgraph\n";

    // Find all h-cliques
    cout << "Finding all " << h << "-cliques... " << flush;
    vector<vector<int>> hCliques;
    {
        vector<int> temp;
        G.findCliques(h, temp, 0, hCliques);
    }
    cout << " Found " << hCliques.size() << " " << h << "-cliques.\n";

    cout << "Finding all " << (h - 1) << "-cliques... " << flush;
    vector<vector<int>> hMinusOneCliqueGroups;
    {
        vector<int> temp;
        G.findCliques(h - 1, temp, 0, hMinusOneCliqueGroups);
    }
    cout << " Found " << hMinusOneCliqueGroups.size() << " " << (h - 1) << "-cliques.\n";

    double l = 0;
    double u = G.findMaxCliqueDegree(h);

    vector<int> D;

    // Binary search for optimal density
    int iterationCounter = 0;
    cout << "Binary search progress: " << flush;

    while (u - l >= 1.0 / (n * (n - 1)))
    {
        iterationCounter++;
        double progress = (u - l) / u * 100.0;

        if (iterationCounter % 5 == 0)
        {
            cout << "\rBinary search: " << fixed << setprecision(1) << (100.0 - progress) << "% (α=" << l << ".." << u << ") " << flush;
        }

        double al = (l + u) / 2;

        cout << "\nBuilding flow network for α=" << al << "... " << flush;
        int numNodes = 1 + n + hMinusOneCliqueGroups.size() + 1; // s + vertices + (h-1)-cliques + t
        vector<vector<int>> capacity(numNodes, vector<int>(numNodes, 0));

        int s = 0;
        int t = numNodes - 1;

        // Add edges from s to vertices
        for (int v = 0; v < n; v++)
        {
            capacity[s][v + 1] = G.cliqueDegree(v, h);
        }

        // Add edges from vertices to t
        for (int v = 0; v < n; v++)
        {
            capacity[v + 1][t] = al * h;
        }

        cout << "Processing edges between vertices and " << (h - 1) << "-cliques... " << flush;
        int edgeCounter = 0;
        const auto &adj = G.getAdjList();

        for (size_t i = 0; i < hMinusOneCliqueGroups.size(); i++)
        {
            for (int v = 0; v < n; v++)
            {
                edgeCounter++;
                if (edgeCounter % 100000 == 0)
                {
                    cout << "." << flush;
                }

                bool canFormClique = true;
                for (int u : hMinusOneCliqueGroups[i])
                {
                    if (v == u)
                        continue;
                    if (find(adj[v].begin(), adj[v].end(), u) == adj[v].end())
                    {
                        canFormClique = false;
                        break;
                    }
                }

                if (canFormClique)
                {
                    capacity[v + 1][n + 1 + i] = 1;
                }
            }
        }
        cout << " Done!\n";

        for (size_t i = 0; i < hMinusOneCliqueGroups.size(); i++)
        {
            for (int v : hMinusOneCliqueGroups[i])
            {
                capacity[n + 1 + i][v + 1] = numeric_limits<int>::max();
            }
        }

        // Find min-cut
        vector<int> minCut;
        fordFulkerson(capacity, s, t, minCut);

        if (minCut.size() <= 1)
        { // Only s is in the cut
            u = al;
            cout << "Cut contains only source. Reducing upper bound to " << u << endl;
        }
        else
        {
            l = al;

            D.clear();
            for (int node : minCut)
            {
                if (node != s && node < n + 1)
                {
                    D.push_back(node - 1); // Convert back to original vertex index
                }
            }
            cout << "Cut contains " << D.size() << " vertices. Increasing lower bound to " << l << endl;
        }
    }

    cout << "\nBinary search complete. Final density: " << l << endl;

    return G.getInducedSubgraph(D);
}

// trying to implementation of the kClist++ algorithm from the paper
vector<double> evaluateKCliqueRanking(const Graph &G, int k, int T)
{
    int n = G.retrieveVertexCount();
    vector<double> r(n, 0.0);

    cout << "Running kClist++ algorithm... " << flush;
    for (int t = 0; t < T; t++)
    {
        if (t % (T / 20) == 0)
        {
            cout << "." << flush;
        }

        // Process each k-clique
        vector<vector<int>> kCliqueGroups;
        vector<int> temp;
        G.findCliques(k, temp, 0, kCliqueGroups);

        for (const auto &C : kCliqueGroups)
        {
            // Find vertex with minimum score
            int minVertex = C[0];
            for (int v : C)
            {
                if (r[v] < r[minVertex])
                {
                    minVertex = v;
                }
            }
            r[minVertex]++;
        }
    }

    // Normalize scores
    for (int u = 0; u < n; u++)
    {
        r[u] /= T;
    }

    cout << " Done!\n";
    return r;
}

int main()
{
   
    ifstream inputFile;
    cout << "Reading input...\n";

    string filename = "net.txt";
    inputFile.open(filename);

    if (!inputFile.is_open())
    {
        cout << "File not found, reading from stdin...\n" ;

        ofstream tempFile("temp_input.txt");
        string line;
        while (getline(cin, line))
        {
            tempFile << line << endl;
        }
        tempFile.close();

        inputFile.open("temp_input.txt");
        if (!inputFile.is_open())
        {
            cerr << "Error creating temporary file!\n";
            return 1;
        }
    }

    int n, m, h;
    inputFile >> n >> m >> h;

    cout << "Creating graph with " << n << " vertices and " << m << " edges...\n";
    Graph G(n);

    for (int i = 0; i < m; i++)
    {
        int u, v;
        inputFile >> u >> v;
        G.addEdge(u, v);


        if (m > 10000 && i % (m / 100) == 0)
        {
            cout << "\rReading edges: " << (i * 100 / m) << "% complete" << flush;
        }
    }
    inputFile.close();

    if (m > 10000)
        cout << "\rReading edges: 100% complete\n" ;

    cout << "Original Graph has " << n << " vertices and " << m << " edges.\n" ;
    cout << "Looking for " << h << "-clique densest subgraph...\n" ;

    // Start time tracking
    auto startTime = chrono::high_resolution_clock::now();

    Graph D = locateCliqueDensitySubgraph(G, h);

    // End time tracking
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();

    cout << "\nCompleted in " << duration << " seconds!\n";
    cout << "Clique-Dense Subgraph found with " << D.retrieveVertexCount() << " vertices!\n";
    cout << "Number of " << h << "-cliques in CDS: " << D.countCliques(h) << endl;
    cout << h << "-clique density of CDS: " << D.cliqueDesity(h) << endl;

    return 0;
}
