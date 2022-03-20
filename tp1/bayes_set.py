
class Naive:
    
    def train(self, x_train, t_train, x_test, t_test):
        t_train_list = list(map(lambda e: e[1], t_train.items()))

        t_counts = {}
        t_probs = {}

        ## Count ocurrences
        for t in t_train_list:
            if t not in t_counts:
                t_counts[t] = 0
            t_counts[t] += 1

        ## Probabilities of T
        t_total = len(t_probs.keys())
        entries_total = len(t_train)
        for t in t_counts:
            ## Laplace Correction
            t_probs[t] = (t_counts[t] + 1) / (entries_total + t_total)

        ### {
        #   x_name: {0, 1}
        # }
        cond_values = {}

        ## x conditional probabilities (x_name=x_i | T=t_i)
        ### {
        #   (x_name, x_i, t_i): prob,
        # }
        cond_probs = {}

        for col_name, content in x_train.items():
            for i, x in content.items():
                t = t_train_list[i]

                ## Ocurrences with (x_name, t) = {0, 1}
                counts_k = col_name
                if counts_k not in cond_values:
                    cond_values[counts_k] = set()
                cond_values[counts_k].add(x)

                prob_k = (col_name, x, t)
                if prob_k not in cond_probs:
                    cond_probs[prob_k] = 0
                cond_probs[prob_k] += 1

        for prob_k in cond_probs:
            (col_name, x, t) = prob_k
            ## Laplace Correction
            cond_probs[prob_k] = (cond_probs[prob_k] + 1) / (t_counts[t] + len(cond_values[col_name])) ## Replace by 2 if needed just 0 or 1 always

        # ## Transform
        # cond_probs_printeable = {}

        # for k in cond_probs:
        #     k_str = str(k)
        #     cond_probs_printeable[k_str] = cond_probs[k]

        # print(json.dumps(cond_probs_printeable, sort_keys=True, indent=2))
        self.t_probs = t_probs
        self.cond_probs = cond_probs
        self.t_counts = t_counts
        self.cond_values = cond_values

        #TODO: return error
        err = 0
        return err

    def eval(self, entries):
        t_probs = self.t_probs
        cond_probs = self.cond_probs
        t_counts = self.t_counts
        cond_values = self.cond_values

        results = []

        x_names = list(entries.columns)
        for i, row in entries.iterrows():
            VNB_t = None
            VNB = 0
            total_prob = 0
            for t in t_probs:
                t_prob = t_probs[t]
                prod = 1
                ## P(1,0,0,1 | t1) * P(t1) = P(1,0,0,1,t1)
                ## prod * t_prob = prob
                for x_name in x_names:
                    x_val = row[x_name]
                    prob_k = (x_name, x_val, t)
                    cond = cond_probs.get(prob_k, 1 / (t_counts[t] + len(cond_values[x_name]))) ## Laplace correction (replace len by 2 if needed)
                    prod *= cond
                
                prob = prod * t_prob
                total_prob += prob

                if prod > VNB:
                    VNB_t = t
                    VNB = prob
            
            results.append((VNB_t, VNB/total_prob))

        return results
        
