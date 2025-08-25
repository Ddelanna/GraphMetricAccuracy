import CreateData
import HelperFunctions
import PredictionMethods
import SamplingAlgorithms
from HelperFunctions import AdjacencyMatrices


def kmeansplusplus(X,labels, W_gaussian, n_clusters, distance_type='peikonal', n_local_trials=None, random_state=None):
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    sample_weight = np.ones(X.shape[0])
    n_samples, n_features = X.shape
    if n_local_trials is None:
        n_local_trials = 3 + int(np.log(n_clusters))
    centers = np.empty((n_clusters, n_features))
    indices = np.full(n_clusters, -1, dtype=int)
    center_id = random_state.integers(n_samples)
    W=W_gaussian
    coreset = [center_id]  # initialize with first center
    centers[0] = X[center_id]
    indices[0] = center_id
    def plot_prop(prop_vals, show_source=True):
        #assert prop_vals.size == n
        fig, ax = plt.subplots()
        p = ax.scatter(X[:,0], X[:,1], c=prop_vals)
        if show_source:
            ax.scatter(X[coreset,0], X[coreset,1], c='r', s=80, marker='^')
        plt.colorbar(p, ax=ax)
        plt.show()
    def compute_distance( W, bdy_set):
        if distance_type == 'dijkstra':
            #W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
            G = gl.graph(W)
            dist = G.dijkstra(bdy_set=bdy_set, bdy_val=0)
            return dist
        elif distance_type == 'peikonal':
            G = gl.graph(W)
            dist = G.peikonal(bdy_set=bdy_set, p=1)
            return dist
        elif distance_type == 'fermat':
            p=1
            #W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
            W_powered = W.copy()
            W_powered **= p
            #W_powered.data = np.power(W_powered.data, p, dtype=float)
            W = W_powered
            G = gl.graph(W)
            dist = G.dijkstra(bdy_set=bdy_set, bdy_val=0)
            return dist
        elif distance_type == 'euclidean':
            dist = W
            return dist
        else:
            raise ValueError(f"Unsupported distance method: {distance_type}")
    dists = compute_distance( W, coreset)
    dists= dists**2
    current_pot = np.sum(dists)
    for c in range(1, n_clusters):
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidates_samples = np.searchsorted(stable_cumsum( dists), rand_vals)
        np.clip(candidates_samples, None, dists.size - 1, out=candidates_samples)
        #candidates_samples = np.random.choice(n_samples, n_local_trials, p=probs, replace=False)
        sums =[]
        dists_per_candidate = []
        for s in candidates_samples:
            bdy_set =   indices[:c].tolist()+[s]# Combine candidate and previous centers
            d = compute_distance( W,bdy_set)
            sums.append(np.sum(d))
            dists_per_candidate.append(d)
        min_index = np.argmin(sums)
        best_sample_point = candidates_samples[min_index]
        dists = dists_per_candidate[min_index]
        dists= dists**2
        current_pot = np.sum(dists)
        centers[c] = X[best_sample_point]
        indices[c] = best_sample_point
    return centers, indices

class Iterate:
    def __init__(self, data_generation, sampling_method, prediction_method, random_state=None):
        NUM_POINTS = 1000
        BUDGET = 10
        RADIUS = 1.5

        unlabeled_points, oracle = data_generation(num_points=NUM_POINTS, random_state=random_state)

        for adjacency_matrix in [AdjacencyMatrices().binary_epsilon_graph(unlabeled_points, RADIUS)]:
            model = sampling_method(unlabeled_points, BUDGET, adjacency_matrix=adjacency_matrix, random_state=random_state)
            score = prediction_method(model, oracle).score
            print('score', score)

if __name__ == '__main__':
    # todo: metrics to use: euclidean, fermat (p=1,2,4?)
    # todo: fix AdjacencyMatrices
    # todo: switch sampling methods over to sparse / improve runtime

    # todo: using MNIST, FashionMNIST, and CIFAR from jeff's graphlearning github
    # todo: record runtimes for creating distance / adjacency matrices and for the algorithm

    Iterate(data_generation=CreateData.create_spiral_data,
            sampling_method=SamplingAlgorithms.ConnectedComponentSampling,
            prediction_method=PredictionMethods.GraphMetricAccuracy,
            random_state=1)

    # THIS BLOCK OF CODE SHOULD RUN NOW WHEN UNCOMMENTED
    # data, oracle = CreateData.create_spiral_data(100)
    # budget = 6
    # distance_matrix = AdjacencyMatrices().distance_matrix(data)
    # print(SamplingAlgorithms.KmeansSampling(data, budget, distance_matrix, random_state=5).query_indices)
    # print(temp.kmeansplusplus(X=np.array(data), labels=oracle, W_gaussian=distance_matrix, n_clusters=budget, random_state=5)[1])







