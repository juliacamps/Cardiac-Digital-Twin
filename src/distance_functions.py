import math
import numpy as np

from utils import insert_sorted, get_nan_value


def djikstra(source_id_list, djikstra_nodes_xyz, djikstra_unfoldedEdges, djikstra_edgeVEC, djikstra_neighbours,
             approx_max_path_len):
    """approx_max_path_len: this parameter is not a hard constraint, but an approximate value to speed up the method.
    In practice, the method will increase its own approx_max_path_len value until it has explored the whole range of
    possibilities."""
    distances = np.zeros((djikstra_nodes_xyz.shape[0], source_id_list.shape[0])).astype(float)
    paths = np.full((djikstra_nodes_xyz.shape[0], source_id_list.shape[0], approx_max_path_len), get_nan_value(), np.int32)
    for source_id_index in range(source_id_list.shape[0]):
        source_id = source_id_list[source_id_index]
        distances_per_source = np.zeros((djikstra_nodes_xyz.shape[0]))
        previous_node_indexes_temp = np.zeros((djikstra_nodes_xyz.shape[0])).astype(
            int)  # 2022/01/10 This object starts with a wrong solution and it makes it better and better until in the end the
        # solution is the correct one. Idea by me and Jenny :-)
        visitedNodes = np.zeros((djikstra_nodes_xyz.shape[0])).astype(bool)

        # Compute the cost of all endocardial edges
        navigationCosts = np.zeros((int(djikstra_unfoldedEdges.shape[0] / 2)))
        for index in range(0, navigationCosts.shape[0]):
            # Cost for the propagation in the endocardium
            navigationCosts[index] = math.sqrt(np.dot(djikstra_edgeVEC[index, :], djikstra_edgeVEC[index, :]))

        # Build adjacentcy costs
        # TODO remove this:
        # if True:
        # for i in range(0, djikstra_nodes_xyz.shape[0], 1):
        #     print('djikstra_neighbours ', djikstra_neighbours)
        adjacentCost = [np.concatenate((djikstra_unfoldedEdges[djikstra_neighbours[i]][:, 1][:, np.newaxis],
                                        navigationCosts[djikstra_neighbours[i] % navigationCosts.shape[0]][:,
                                        np.newaxis]), axis=1) for i in range(0, djikstra_nodes_xyz.shape[0], 1)]

        cummCost = 0.  # Distance from a node to itself is zero
        tempDists = np.zeros((djikstra_nodes_xyz.shape[0],), float) + 1000

        ## Run the code for the root nodes
        visitedNodes[source_id] = True
        distances_per_source[source_id] = cummCost
        nextNodes = (adjacentCost[source_id] + np.array([0, cummCost])).tolist()
        activeNode_i = source_id
        sortSecond = lambda x: x[1]
        nextNodes.sort(key=sortSecond, reverse=True)
        previous_node_indexes_temp[activeNode_i] = activeNode_i  # 2022/01/10
        for nextEdge_aux in nextNodes:  # 2022/01/10
            previous_node_indexes_temp[int(nextEdge_aux[0])] = activeNode_i  # 2022/01/10
        while visitedNodes[activeNode_i] and len(nextNodes) > 0:
            nextEdge = nextNodes.pop()
            activeNode_i = int(nextEdge[0])
        cummCost = nextEdge[1]
        if nextNodes:  # Check if the list is empty, which can happen while everything being Ok
            tempDists[(np.array(nextNodes)[:, 0]).astype(int)] = np.array(nextNodes)[:, 1]

        ## Run the whole algorithm
        for i in range(distances_per_source.shape[0]):
            visitedNodes[activeNode_i] = True
            distances_per_source[activeNode_i] = cummCost
            adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist()
            for adjacent_i in range(0, len(adjacents), 1):
                if (not visitedNodes[int(adjacents[adjacent_i][0])] and (
                        tempDists[int(adjacents[adjacent_i][0])] > adjacents[adjacent_i][1])):
                    insert_sorted(nextNodes, adjacents[adjacent_i])
                    tempDists[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
                    previous_node_indexes_temp[int(adjacents[adjacent_i][0])] = activeNode_i  # 2022/01/10
            while visitedNodes[activeNode_i] and len(nextNodes) > 0:
                nextEdge = nextNodes.pop()
                activeNode_i = int(nextEdge[0])
            cummCost = nextEdge[1]

        distances[:, source_id_index] = distances_per_source
        for djikstra_node_id in range(0, djikstra_nodes_xyz.shape[0], 1):  # 2022/01/10
            path_per_source = np.full((djikstra_nodes_xyz.shape[0]), get_nan_value(), np.int32)  # 2022/01/10
            path_node_id = djikstra_node_id  # 2022/01/10
            path_node_id_iter = 0  # 2022/01/10
            path_per_source[path_node_id_iter] = path_node_id  # 2022/01/14
            path_node_id_iter = path_node_id_iter + 1  # 2022/01/14
            while path_node_id != source_id:  # 2022/01/10
                path_node_id = previous_node_indexes_temp[path_node_id]  # 2022/01/10
                path_per_source[path_node_id_iter] = path_node_id  # 2022/01/10
                path_node_id_iter = path_node_id_iter + 1  # 2022/01/10
            # If the path is longer than the current size of the matrix, make the matrix a little bigger and continue
            if path_node_id_iter + 1 > approx_max_path_len:  # 2022/01/11
                paths_aux = np.full((djikstra_nodes_xyz.shape[0], source_id_list.shape[0], path_node_id_iter + 10),
                                    get_nan_value(), np.int32)  # 2022/01/11
                paths_aux[:, :, :approx_max_path_len] = paths  # 2022/01/11
                paths = paths_aux  # 2022/01/11
                approx_max_path_len = path_node_id_iter + 10  # 2022/01/11
            paths[djikstra_node_id, source_id_index, :] = path_per_source[:approx_max_path_len]  # 2022/01/10
    return distances, paths  # 2022/01/
