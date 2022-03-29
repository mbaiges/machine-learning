import pandas as pd

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
        t_total = len(t_counts.keys())
        entries_total = len(t_train)
        for t in t_counts:
            ## Laplace Correction
            t_probs[t] = (t_counts[t] + 1) / (entries_total + t_total)

        print("---------------- PROBS ----------------")
        for class_, prob in t_probs.items():
            print(f"P({class_}) = {prob}")
        print("---------------------------------------")
        ## x conditional probabilities (x_name=x_i | T=t_i)
        ### {
        #   (x_name, x_i, t_i): prob,
        # }
        cond_probs = {}

        for col_name, content in x_train.items(): ## cada columna
            for i, x in content.items(): ## cada fila de esa columna
                t = t_train_list[i] ## etiqueta de la fila

                prob_k = (col_name, x, t)
                if prob_k not in cond_probs:
                    cond_probs[prob_k] = 0
                cond_probs[prob_k] += 1

        for prob_k in cond_probs:
            (col_name, x, t) = prob_k
            ## Laplace Correction
            cond_probs[prob_k] = (cond_probs[prob_k] + 1) / (t_counts[t] + 2)

        print("---------------- COND PROBS ----------------")
        for prob_k, prob in cond_probs.items():
            (col_name, x, t) = prob_k
            print(f"P({col_name} = {x} | class = {t}) = {prob}")
        print("--------------------------------------------")
        # ## Transform
        # cond_probs_printeable = {}

        # for k in cond_probs:
        #     k_str = str(k)
        #     cond_probs_printeable[k_str] = cond_probs[k]

        # print(json.dumps(cond_probs_printeable, sort_keys=True, indent=2))
        self.t_probs = t_probs
        self.cond_probs = cond_probs
        self.t_counts = t_counts

        ret = None
        if x_test:
            err = 0
            results = self.eval(x_test)
            for res, expected in zip(results, t_test):
                cat = res[0]
                if(res != expected):
                    err += 1
            ret = err/len(x_test)
        return ret

    def eval(self, entries):
        t_probs = self.t_probs
        cond_probs = self.cond_probs
        t_counts = self.t_counts

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
                    cond = cond_probs.get(prob_k, 1 / (t_counts[t] + 2)) ## Laplace correction (replace len by 2 if needed)
                    prod *= cond
                
                prob = prod * t_prob
                total_prob += prob

                if prod > VNB:
                    VNB_t = t
                    VNB = prob
            
            results.append((VNB_t, VNB/total_prob))

        return results
        
