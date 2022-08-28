import math
from typing import Optional
from autogoal.utils import Min, Gb, Sec
from ._pge import PESearch


class NSPESearch(PESearch):
    def __init__(
        self,
        # =============================
        # args for SearchAlgorithm
        # =============================
        generator_fn=None,
        fitness_fn=None,
        pop_size=None,
        maximize=True,
        errors="raise",
        early_stop=0.5,
        evaluation_timeout: int = 10 * Sec,
        memory_limit: int = 4 * Gb,
        search_timeout: int = 5 * Min,
        target_fn=None,
        allow_duplicates=True,
        number_of_solutions=None,
        ranking_fn=None,
        # =============================
        # args for PESearch
        # =============================
        learning_factor: float = 0.05,
        selection: float = 0.2,
        epsilon_greed: float = 0.1,
        random_state: Optional[int] = None,
        name: str = None,
        save: bool = False,
        # =============================
        # others
        # =============================
        **kwargs
    ):
        def default_ranking_fn(_, fns):
            # use worst possible ranking by default, although all rankings should be
            # set to a number by the end of the function
            rankings = [-math.inf] * len(fns)
            fronts = self.non_dominated_sort(fns)
            for ranking, front in enumerate(fronts):
                for index in front:
                    # the closest you are to front 0 the highest your ranking
                    rankings[index] = -ranking
            return rankings

        if ranking_fn is None:
            ranking_fn = default_ranking_fn

        super().__init__(
            generator_fn=generator_fn,
            fitness_fn=fitness_fn,
            pop_size=pop_size,
            maximize=maximize,
            errors=errors,
            early_stop=early_stop,
            evaluation_timeout=evaluation_timeout,
            memory_limit=memory_limit,
            search_timeout=search_timeout,
            target_fn=target_fn,
            allow_duplicates=allow_duplicates,
            number_of_solutions=number_of_solutions,
            ranking_fn=ranking_fn,
            learning_factor=learning_factor,
            selection=selection,
            epsilon_greed=epsilon_greed,
            random_state=random_state,
            name=name,
            save=save,
            **kwargs
        )

    def _indices_of_fittest(self, fns):
        fronts = self.non_dominated_sort(fns)
        indices, k = [], int(self._selection * len(fns))
        for front in fronts:
            if len(indices) + len(front) <= k:
                indices.extend(front)
            else:
                indices.extend(
                    sorted(
                        front,
                        key=lambda i: (fns[i], self.crowding_distance(fns, front, i)),
                        reverse=True,
                    )[: k - len(indices)]
                )
                break
        return indices

    def non_dominated_sort(self, fns):
        fronts = [[]]
        domination_counts = [0] * len(fns)
        dominated_fns = [[] for _ in fns]

        for i, fn_i in enumerate(fns):
            for j, fn_j in enumerate(fns):
                if self._improves(fn_i, fn_j):
                    dominated_fns[i].append(j)
                elif self._improves(fn_j, fn_i):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for idx in fronts[i]:
                for dominated_idx in dominated_fns[idx]:
                    domination_counts[dominated_idx] -= 1
                    if domination_counts[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance(self, fns, front, i):
        if len(front) == 0:
            raise ValueError("Front is empty.")
        elif len(front) < 0:
            raise ValueError("Front is negative.")

        crowding_distances = [0] * len(fns)
        for m in range(len(self._maximize)):
            front = list(sorted(front, key=lambda i: fns[i][m]))
            crowding_distances[front[0]] = math.inf
            crowding_distances[front[-1]] = math.inf
            m_values = [fns[i][m] for i in front]
            scale = max(m_values) - min(m_values)
            if scale == 0:
                scale = 1
            for i in range(1, len(front) - 1):
                crowding_distances[i] += (
                    fns[front[i + 1]][m] - fns[front[i - 1]][m]
                ) / scale
        return crowding_distances[i]
