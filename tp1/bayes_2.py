import math

class Naive:
    
    def train(self, x_train, t_train, x_test, t_test):
        t_train_list = t_train

        t_counts = {}
        t_probs = {}

        # Tag

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

        # Conditional Probabilities

        words_by_category = {}

        ## x conditional probabilities (x_name=x_i | T=t_i)
        ### {
        #   (x_name, x_i, t_i): prob,
        # }
        cond_probs = {}

        for i, bag in enumerate(x_train): ## Por cada bag (noticia)
            for w, total in bag.items(): ## Por cada palabra en esa noticia
                t = t_train_list[i] ## Etiqueta de esa noticia

                prob_k = (w, 1, t)
                if prob_k not in cond_probs:
                    cond_probs[prob_k] = 0
                cond_probs[prob_k] += total

                if t not in words_by_category:
                    words_by_category[t] = 0
                words_by_category[t] += total

        for prob_k in cond_probs:
            (col_name, x, t) = prob_k
            ## Laplace Correction
            cond_probs[prob_k] = (cond_probs[prob_k] + 1) / (words_by_category[t] + 2) ## Replace by 2 if needed just 0 or 1 always

        # ## Transform
        # cond_probs_printeable = {}

        # for k in cond_probs:
        #     k_str = str(k)
        #     cond_probs_printeable[k_str] = cond_probs[k]

        # print(json.dumps(cond_probs_printeable, sort_keys=True, indent=2))
        self.t_probs = t_probs
        self.cond_probs = cond_probs
        self.t_counts = t_counts
        self.words_by_category = words_by_category

        ret = None
        if x_test:
            err = 0
            results = Naive.get_most_probable_class(self.eval(x_test))
            for (cat, prob), expected in zip(results, t_test):
                if cat != expected:
                    err += 1
            ret = err/len(x_test)
        return ret

    def eval(self, entries):
        t_probs = self.t_probs
        cond_probs = self.cond_probs
        t_counts = self.t_counts
        words_by_category = self.words_by_category

        results = []

        for i, bag in enumerate(entries):
            VNB_t = None
            VNB = - math.inf
            total_prob = 0
            all_probs = {}
            for t in t_probs:
                t_prob = t_probs[t]
                prod = 1
                ## P(1,0,0,1 | t1) * P(t1) = P(1,0,0,1,t1)
                ## prod * t_prob = prob
                # print(cond_probs)
                for x_name in bag:
                    present = 1
                    prob_k = (x_name, present, t)
                    cond = cond_probs.get(prob_k, 1 / (words_by_category[t] + 2)) ## Laplace correction (replace len by 2 if needed)
                    prod *= cond
                # for x_name in bag:
                #     x_val = bag[x_name]
                #     prob_k = (x_name, 1, t)
                #     cond = cond_probs.get(prob_k, 1 / (words_by_category[t] + 2)) ## Laplace correction (replace len by 2 if needed)
                #     prod *= cond
                prob = prod * t_prob
                total_prob += prob
                # print(total_prob)
                all_probs[t] = prob
                # if prod > VNB:
                #     VNB_t = t
                #     VNB = prob
            
            for t in all_probs:
                all_probs[t] = all_probs[t] / total_prob

            results.append(all_probs) 

        return results

    @staticmethod
    def get_prob_for_class(result, clazz):
        return result[clazz]
    
    @staticmethod      
    def get_most_probable_class(results):
        ret = []
        for result in results:
            c, p = None, None
            m = - math.inf
            for clazz in result:
                prob = Naive.get_prob_for_class(result, clazz)
                if prob > m:
                    c = clazz
                    p = m = prob
            ret.append((c, p))
        return ret
