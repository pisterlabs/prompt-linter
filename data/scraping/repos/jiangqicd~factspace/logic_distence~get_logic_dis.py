from utils.constants import coherence
from utils.constants import time


class logic_dis_extractor:

    def __init__(self, vl_specs):
        self.vl_specs = vl_specs

    def logic_detect(self):

        distence_matrix = []
        for vm in self.vl_specs:
            dis = []
            for vn in self.vl_specs:
                logic = {"rs": 0,
                         "rt": 0,
                         "rc": 0,
                         "re": 0,
                         "rg": 0, }
                Vm = self.vl_specs[vm]
                Vn = self.vl_specs[vn]
                if vm != vn:
                    if self.rs_detect(Vm["vis"], Vn["vis"]) == 1:
                        logic["rs"] = 1
                    elif self.rt_detect(Vm["vis"], Vn["vis"]) == 1:
                        logic["rt"] = 1
                    elif self.rc_detect(Vm["vis"], Vn["vis"]) == 1:
                        logic["rc"] = 1
                    elif self.re_detect(Vm["vis"], Vn["vis"]) == 1:
                        logic["re"] = 1
                    elif self.rg_detect(Vm["vis"], Vn["vis"]) == 1:
                        logic["rg"] = 1
                task = Vm["task"]
                d = 0
                for k in coherence[task]:
                    if k != "ra":
                        d += coherence[task][k] * logic[k]
                dis.append(d)
            distence_matrix.append(dis)
        return distence_matrix

    # Similarity indicates two succeeding facts are logically parallel to each other.
    # Both subspace and observation attributes are the same
    # Aggregation of observed attributes makes a difference

    def rs_detect(self, m, n):

        m_filters = []
        n_filters = []
        m_transform = {}
        n_transform = {}

        if "transform" in m:
            m_transform = m["transform"][0]
        if "transform" in n:
            n_transform = n["transform"][0]

        if "filter" in m_transform:
            if "and" in m_transform["filter"]:
                m_filters.extend(m_transform["filter"]["and"])
            else:
                m_filters.append(m_transform["filter"])

        if "filter" in n_transform:
            if "and" in n_transform["filter"]:
                n_filters.extend(n_transform["filter"]["and"])
            else:
                n_filters.append(n_transform["filter"])
        m = 0
        n = 0
        for f in m_filters:
            if f in n_filters:
                m += 1
        for f in n_filters:
            if f in m_filters:
                n += 1

        if m == len(m_filters) and n == len(n_filters):
            return 1
        else:
            return 0
        # m_encoding = {}
        # n_encoding = {}
        #
        # if "encoding" in m:
        #     m_encoding = m["encoding"]
        # if "encoding" in n:
        #     n_encoding = n["encoding"]
        #
        # if n_filters != m_filters:
        #     return 0
        # else:
        #     if list(m_encoding.keys()) != list(n_encoding.keys()):
        #         return 0
        #     else:
        #         flag = 0
        #         for k in m_encoding.keys():
        #             m_field = m_encoding[k]["field"]
        #             n_field = n_encoding[k]["field"]
        #             m_aggregate = m_encoding[k]["aggregate"]
        #             n_aggregate = n_encoding[k]["aggregate"]
        #             if m_field != n_field:
        #                 return 0
        #             else:
        #                 if m_aggregate != n_aggregate:
        #                     flag = 1
        #         if flag == 1:
        #             return 1
        #         else:
        #             return 0

    # Temporal relation communicates the ordering in time of events or states. In this case,
    # we generate f(i+1) by changing the value of the temporal filter in f(i)’s subspace to a succeeding time.
    def rt_detect(self, m, n):
        m_filters = []
        n_filters = []
        m_transform = {}
        n_transform = {}

        if "transform" in m:
            m_transform = m["transform"][0]
        if "transform" in n:
            n_transform = n["transform"][0]

        if "filter" in m_transform:
            if "and" in m_transform["filter"]:
                m_filters.extend(m_transform["filter"]["and"])
            else:
                m_filters.append(m_transform["filter"])

        if "filter" in n_transform:
            if "and" in n_transform["filter"]:
                n_filters.extend(n_transform["filter"]["and"])
            else:
                n_filters.append(n_transform["filter"])

        # to check if there is a timing relationship
        mt_filters = []
        mo_filters = []
        nt_filters = []
        no_filters = []

        for f in m_filters:
            if f["field"].lower() not in time:
                mo_filters.append(f)
            else:
                mt_filters.append(f)

        for f in n_filters:
            if f["field"].lower() not in time:
                no_filters.append(f)
            else:
                nt_filters.append(f)

        m = 0
        n = 0
        for f in mt_filters:
            if f in nt_filters:
                m += 1
        for f in nt_filters:
            if f in mt_filters:
                n += 1

        if m == len(mt_filters) and n == len(nt_filters):
            return 1
        else:
            return 0
        # # chronological order is not considered now
        # if mo_filters != no_filters:
        #     return 0
        # else:
        #     if len(mt_filters) == 1 and len(nt_filters) == 1:
        #         return 1
        #     else:
        #         return 0

    # Contrast indicates a contradiction between two facts. For simplicity,
    # we only check the contradictions in two types of facts, i.e., trend
    # and association. f(i+1) is generated by modifying the subspace of f(i)
    # to form a data contradiction in measures.
    def rc_detect(self, m, n):

        # m_filters = []
        # n_filters = []
        # m_transform = {}
        # n_transform = {}
        #
        # if "transform" in m:
        #     m_transform = m["transform"][0]
        # if "transform" in n:
        #     n_transform = n["transform"][0]
        #
        # if "filter" in m_transform:
        #     if "and" in m_transform["filter"]:
        #         m_filters.extend(m_transform["filter"]["and"])
        #     else:
        #         m_filters.append(m_transform["filter"])
        #
        # if "filter" in n_transform:
        #     if "and" in n_transform["filter"]:
        #         n_filters.extend(n_transform["filter"]["and"])
        #     else:
        #         n_filters.append(n_transform["filter"])

        m_encoding = {}
        n_encoding = {}

        if "encoding" in m:
            m_encoding = m["encoding"]
        if "encoding" in n:
            n_encoding = n["encoding"]

        m_fields = []
        n_fields = []
        for k in m_encoding.keys():
            m_fields.append(m_encoding[k]["field"])
        for k in n_encoding.keys():
            n_fields.append(n_encoding[k]["field"])

        m = 0
        n = 0
        for f in m_fields:
            if f in n_fields:
                m += 1
        for f in n_fields:
            if f in m_fields:
                n += 1

        if m == len(m_fields) and n == len(n_fields):
            return 1
        else:
            return 0

        # if m_fields != n_fields:
        #     return 0
        # else:
        #     if len(m_filters) != len(n_filters):
        #         return 0
        #     else:
        #         m_filters_fileds = []
        #         n_filter_fileds = []
        #         m_filters_values = []
        #         n_filter_values = []
        #         for f in m_filters:
        #             m_filters_fileds.append(f["field"])
        #             m_filters_values.append(f["oneOf"])
        #         for f in n_filters:
        #             n_filter_fileds.append(f["field"])
        #             n_filter_values.append(f["oneOf"])
        #         if m_filters_fileds == n_filter_fileds and m_filters_values != n_filter_values:
        #             return 1
        #         else:
        #             return 0

    # Cause-Effect indicates the later event is caused by the former one. In
    # multidimensional data, a causal relation can be determined between
    # dimensions based on the data distribution. In this way, f(i+1) can
    # be generated by changing the measure field m(i) of f(i)
    # to another numerical field in the spreadsheet that is most likely caused by m(i)
    # in accordance with causal analysis
    def ra_detect(self, m, n):

        m_filters = []
        n_filters = []
        m_transform = {}
        n_transform = {}

        if "transform" in m:
            m_transform = m["transform"][0]
        if "transform" in n:
            n_transform = n["transform"][0]

        if "filter" in m_transform:
            if "and" in m_transform["filter"]:
                m_filters.extend(m_transform["filter"]["and"])
            else:
                m_filters.append(m_transform["filter"])

        if "filter" in n_transform:
            if "and" in n_transform["filter"]:
                n_filters.extend(n_transform["filter"]["and"])
            else:
                n_filters.append(n_transform["filter"])

        m_encoding = {}
        n_encoding = {}

        if "encoding" in m:
            m_encoding = m["encoding"]
        if "encoding" in n:
            n_encoding = n["encoding"]

        if n_filters != m_filters:
            return 0
        else:
            if list(m_encoding.keys()) != list(n_encoding.keys()):
                return 0
            else:
                flag = 0
                for k in m_encoding.keys():
                    m_field = m_encoding[k]["field"]
                    n_field = n_encoding[k]["field"]
                    m_type = m_encoding[k]["type"]
                    n_type = n_encoding[k]["type"]
                    if m_type == "quantitative" and n_type == "quantitative":
                        if m_field != n_field:
                            flag = 1
                    elif m_type == "nominal" and n_type == "quantitative":
                        return 0
                    elif m_type == "quantitative" and n_type == "nominal":
                        return 0
                    elif m_type == "nominal" and n_type == "nominal":
                        if m_field != n_field:
                            return 0
                if flag == 1:
                    return 1
                else:
                    return 0

            # Elaboration indicates a relation in which a latter fact f(i+1) adds more details to the previous one f(i)

    # Elaboration indicates a relation in which a latter fact f(i+1) adds more
    # details to the previous one f(i). In this way, we create f(i+1) by shrinking
    # the scope of fi’s subspace via adding more constraints (i.e., filters)
    # or setting a focus to “zoom” f(i) into a more specific scope.
    def re_detect(self, m, n):

        m_filters = []
        n_filters = []
        m_transform = {}
        n_transform = {}

        if "transform" in m:
            m_transform = m["transform"][0]
        if "transform" in n:
            n_transform = n["transform"][0]

        if "filter" in m_transform:
            if "and" in m_transform["filter"]:
                m_filters.extend(m_transform["filter"]["and"])
            else:
                m_filters.append(m_transform["filter"])

        if "filter" in n_transform:
            if "and" in n_transform["filter"]:
                n_filters.extend(n_transform["filter"]["and"])
            else:
                n_filters.append(n_transform["filter"])

        m_encoding = {}
        n_encoding = {}

        if "encoding" in m:
            m_encoding = m["encoding"]
        if "encoding" in n:
            n_encoding = n["encoding"]

        m_fields = []
        n_fields = []
        for k in m_encoding.keys():
            m_fields.append(m_encoding[k]["field"])
        for k in n_encoding.keys():
            n_fields.append(n_encoding[k]["field"])

        m = 0
        n = 0
        for f in m_fields:
            if f in n_fields:
                m += 1
        for f in n_fields:
            if f in m_fields:
                n += 1

        if m == len(m_fields) and n == len(n_fields):
            if m_filters == n_filters:
                return 0
            else:
                l = 0
                for f in m_filters:
                    if f in n_filters:
                        l += 1
                if l == len(m_filters):
                    return 1
                else:
                    return 0
        else:
            return 0

        # if m_fields != n_fields:
        #     return 0
        # else:
        #     if m_filters == n_filters:
        #         return 0
        #     else:
        #         l = 0
        #         for f in m_filters:
        #             if f in n_filters:
        #                 l += 1
        #         if l == len(m_filters):
        #             return 1
        #         else:
        #             return 0

    # Generalization indicates f(i) is an abstraction of the previous f(i),which is in opposite to elaboration.
    def rg_detect(self, m, n):

        m_filters = []
        n_filters = []
        m_transform = {}
        n_transform = {}

        if "transform" in m:
            m_transform = m["transform"][0]
        if "transform" in n:
            n_transform = n["transform"][0]

        if "filter" in m_transform:
            if "and" in m_transform["filter"]:
                m_filters.extend(m_transform["filter"]["and"])
            else:
                m_filters.append(m_transform["filter"])

        if "filter" in n_transform:
            if "and" in n_transform["filter"]:
                n_filters.extend(n_transform["filter"]["and"])
            else:
                n_filters.append(n_transform["filter"])

        m_encoding = {}
        n_encoding = {}

        if "encoding" in m:
            m_encoding = m["encoding"]
        if "encoding" in n:
            n_encoding = n["encoding"]

        m_fields = []
        n_fields = []
        for k in m_encoding.keys():
            m_fields.append(m_encoding[k]["field"])
        for k in n_encoding.keys():
            n_fields.append(n_encoding[k]["field"])

        m = 0
        n = 0
        for f in m_fields:
            if f in n_fields:
                m += 1
        for f in n_fields:
            if f in m_fields:
                n += 1

        if m == len(m_fields) and n == len(n_fields):
            if m_filters == n_filters:
                return 0
            else:
                l = 0
                for f in n_filters:
                    if f in m_filters:
                        l += 1
                if l == len(n_filters):
                    return 1
                else:
                    return 0
        else:
            return 0
