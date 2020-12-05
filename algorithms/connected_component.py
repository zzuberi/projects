# Grids of 0s and 1s, 1 representing land 0 representing water
# Replace all the islands with their size
# E.g
# Input:
# [[0, 0, 0, 1, 0],
#  [1, 1, 0, 1, 0],
#  [0, 0, 0, 1, 1],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 1, 0]]
# Output:
# 0 0 0 4 0
# 2 2 0 4 0
# 0 0 0 4 4
# 6 6 0 0 0
# 6 6 6 6 0

# Input:
# [[0, 0, 0, 1, 0],
#  [1, 1, 1, 1, 0],
#  [0, 0, 0, 1, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 1, 0]]

# Input:
# [[1, 1, 1, 1, 1],
#  [0, 0, 0, 0, 1],
#  [1, 1, 1, 0, 1],
#  [1, 0, 0, 0, 1],
#  [1, 1, 1, 1, 1]]
import time
from queue import SimpleQueue
class Islands:
    def connected_land(self, i, j, grid, isle_set):
        if j + 1 < len(grid[0]):  # right
            if grid[i][j + 1] == 1:
                isle_set.add((i, j + 1))
        if i + 1 < len(grid):  # below
            if grid[i + 1][j] == 1:
                isle_set.add((i + 1, j))

    # def island_sizes(self, grid):
    #     islands = []  # list of sets of tuples
    #
    #     for i in range(len(grid)):
    #         for j in range(len(grid[0])):
    #             if grid[i][j] == 1:  # hit island
    #                 isle_set = set()
    #                 candidate_isles = []
    #                 for isle in range(len(islands)):  # check if land is in existing island
    #                     if (i, j) in islands[isle]:
    #                         candidate_isles.append(isle)
    #
    #                 if len(candidate_isles) > 1: #multiple candidates
    #                     for k, isles in enumerate(candidate_isles):
    #                         isle_set |= islands[isles]
    #                         if k > 0:
    #                             del islands[isles]
    #                     islands.append(isle_set)
    #                 elif len(candidate_isles) == 1: #existing island
    #                     isle_set = islands[candidate_isles[0]]
    #                 else:  # new island
    #                     isle_set.add((i, j))
    #                     islands.append(isle_set)
    #                 self.connected_land(i, j, grid, isle_set)
    #
    #     # replace island values with sizes
    #     self.update_grid(grid, islands)
    #     return grid

    # def island_sizes(self, grid):
    #     islands = []
    #     for i in range(len(grid)):
    #         for j in range(len(grid[0])):
    #             if grid[i][j] == 1:
    #                 new_island = set()
    #                 islands.append(new_island)
    #                 self.create_island(grid, i, j, new_island)
    #
    #     self.update_grid(grid, islands)
    #     return grid

    def update_grid(self, grid, islands):
        for isle in islands:
            size = len(isle)
            for coor in isle:
                grid[coor[0]][coor[1]] = size

    # def create_island(self, grid, i, j, island):
    #     if grid[i][j] == 1:
    #         island.add((i,j))
    #         grid[i][j] = -1
    #         if i - 1 >= 0:
    #             self.create_island(grid, i-1, j, island)  # above
    #         if i + 1 < len(grid):
    #             self.create_island(grid, i+1, j, island)  # below
    #         if j - 1 >= 0:
    #             self.create_island(grid, i, j-1, island)  # left
    #         if j + 1 < len(grid[i]):
    #             self.create_island(grid, i, j+1, island)  # right

    def island_sizes(self, grid):
        islands = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    new_island = []
                    islands.append(new_island)
                    self.create_island(grid, i, j, new_island)

        self.update_grid(grid, islands)
        return grid

    def create_island(self, grid, i, j, island):
        to_check = SimpleQueue()
        grid[i][j] = -1
        to_check.put((i, j))
        while not to_check.empty():
            (i, j) = to_check.get()
            island.append((i, j))
            if i - 1 >= 0 and grid[i-1][j] == 1:
                grid[i-1][j] = -1
                to_check.put((i-1, j))
            if i + 1 < len(grid) and grid[i+1][j] == 1:
                grid[i+1][j] = -1
                to_check.put((i+1, j))
            if j - 1 >= 0 and grid[i][j-1] == 1:
                grid[i][j-1] = -1
                to_check.put((i, j-1))
            if j + 1 < len(grid[i]) and grid[i][j+1] == 1:
                grid[i][j+1] = -1
                to_check.put((i, j+1))


if __name__ == "__main__":
    grid = [[1, 1, 1, 1, 1],
 [0, 0, 0, 0, 1],
 [1, 1, 1, 0, 1],
 [1, 0, 0, 0, 1],
 [1, 1, 1, 1, 1]]
    t = time.time()
    out = Islands().island_sizes(grid)
    print(time.time() - t)
    for x in out: print(x)
