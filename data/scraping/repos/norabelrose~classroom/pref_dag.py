from .pref_graph import CoherenceViolation, PrefGraph
import networkx as nx


class PrefDAG(PrefGraph):
    """
    `PrefDAG` enforces the invariant that strict preferences must be transitive, and therefore
    the subgraph representing them must be acyclic. Violating this property will result in a
    `TransitivityViolation` exception, which has a `cycle` attribute that can be used to display
    the offending cycle to the user. We do not assume indifferences are transitive due to the Sorites
    paradox. See <https://en.wikipedia.org/wiki/Sorites_paradox#Resolutions_in_utility_theory>.
    """
    def add_edge(self, a: str, b: str, **attr):
        super().add_edge(a, b, **attr)

        if attr.get('weight', 1) > 0:
            # This is a strict preference, so we should check for cycles
            try:
                cycle = nx.find_cycle(self.strict_prefs, source=a)
            except nx.NetworkXNoCycle:
                pass
            else:
                # Remove the edge we just added.
                self.remove_edge(a, b)

                ex = TransitivityViolation(f"Adding {a} > {b} would create a cycle: {cycle}")
                ex.cycle = cycle
                raise ex
    
    add_pref = add_edge
    
    def add_edges_from(self, ebunch_to_add, **attr):
        # We have to override this method separately since the default implementation doesn't in turn
        # call `add_edge`. As an added bonus we can amortize the cost of checking for coherence violations.
        super().add_edges_from(ebunch_to_add, **attr)

        try:
            cycle = nx.find_cycle(self.strict_prefs)
        except nx.NetworkXNoCycle:
            pass
        else:
            self.remove_edges_from(ebunch_to_add)
            raise TransitivityViolation(f"Edges would create a cycle: {cycle}")
    
    def acyclic_subgraph(self) -> 'PrefDAG':
        return self
    
    def is_quasi_transitive(self) -> bool:
        # The only way that this could end up being false is if someone modifies private
        # attributes of the graph directly in order to create a cycle
        return True
    
    def transitive_closure(self) -> 'PrefDAG':
        """Return a new `PrefDAG` whose strict preference relation is the transitive closure of this one,
        while keeping the indifferences intact."""
        closure = nx.DiGraph(self.strict_prefs)
        order = list(nx.topological_sort(self.strict_prefs))

        # Algorithm copied from `nx.transitive_closure_dag`- we don't use this function directly
        # in order to avoid an unnecessary copy
        for v in reversed(order):
            closure.add_edges_from((v, u) for u in nx.descendants_at_distance(closure, v, 2))
        
        out = PrefDAG(closure)
        out.add_edges_from(self.indifferences.edges(data=True))
        return out

class TransitivityViolation(CoherenceViolation):
    """Raised when a mutation of a `PrefDAG` would cause transitivity to be violated"""
    cycle: list[int]
