# import multiprocessing
from typing import Literal, Tuple, Optional, List
import numpy as np
from .base import Element, Individual
import time
import warnings
from deap import tools
from copy import deepcopy
from .mutations import *
from .crossings import *
from tqdm.auto import tqdm
from IPython.display import clear_output
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
import inspect
warnings.filterwarnings("ignore")

class MGGP:

    def __init__(self,
                 inputs: np.ndarray = np.ndarray((0, 0)),
                 outputs: np.ndarray = np.ndarray((0, 0)),
                 generations: int = 100,
                 validation: Optional[Tuple[np.ndarray, np.ndarray]] = (None, None),
                 evaluationMode: Literal['RMSE', 'MSE', 'MAPE'] = 'RMSE',
                 evaluationType: Literal['OSA', 'MShooting', 'FreeRun', 'INSTANT'] = 'MShooting',
                 evaluationTypeTest: Literal['OSA', 'MShooting', 'FreeRun', 'INSTANT'] = 'FreeRun',
                 k: int = 5,
                 nTerms: int = 15,
                 maxHeight: int = 15,
                 weights: tuple = (-1,),
                 nDelays: int | List[int] = 15,
                 crossoverRate: float = 0.8,
                 mutationRate: float = 0.1,
                 populationSize: int = 100,
                 elitePercentage: int = 10,
                 filename: str = "best_model.pkl",
                 mode: Literal['NARX', 'FIR'] = 'NARX',
                 problem_type: Literal['regression', 'classification'] = 'regression',
                 classification_metric: Literal['accuracy', 'log_loss', 'f1_macro'] = 'accuracy',
                 froe_mode: bool = False, 
                 pruning_probability: float = 0.5,     
                 pruning_tolerance: float = 1e-5,      
                 phi_functions: List[str] = ['subtraction', 'sign'],  
                 operators: List[str] = ['mul'],   
                 **kwargs):
        """
        Args:
            inputs (ndarray): The inputs in the system. Each column represent an input.
            outputs (ndarray): The outputs in the system. Each column represent an output.
            generations (int): Number of generations to train the model.
            validation (Optional[Tuple[np.ndarray, np.ndarray]]): Inputs and outputs to validate the model. Must be a tuple (inputs, outputs).
            evaluationMode (Literal['RMSE', 'MSE', 'MAPE']):  Mode to evaluate the models and ranking the better.
            evaluationType (Literal['OSA', 'MShooting', 'FreeRun']): One-Step-Ahead, Multiple-Shooting and Free-Run predictors.
            evaluationTypeTest (Literal['OSA', 'MShooting', 'FreeRun']): One-Step-Ahead, Multiple-Shooting and Free-Run predictors for Test.
            k (int): Used with Multiple-Shooting predictor. Define the number of shooting.
            nTerms (int): Number of terms each output model will possess.
            maxHeight (int): Maximum height of Genetic Program individual.
            weights (tuple): Defines the type of optimization (-1 for minimization, 1 for maximization). It must be a tuple.
            nDelays (float | Literal['fixed']): The number that will define the backshift operators q^{-n}, for n in delays.
            crossoverRate (float): Crossover probability.
            mutationRate (float): Mutation probability.
            populationSize (int): Population size.
            elitePercentage (int): Percentile of population to be kept in the next generation.
            problem_type (Literal['regression', 'classification']): problem type ('regression' or 'classification')
            classification_metric (Literal['accuracy', 'log_loss', 'f1_macro']): Metric for evaluation in classification problem type
        """

        self.inputs = inputs
        self.outputs = outputs
        self.validation = validation
        self.generations = generations
        self.evaluationMode = evaluationMode
        self.evaluationType = evaluationType
        self.evaluationTypeTest = evaluationTypeTest
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.populationSize = populationSize
        self.elitePercentage = elitePercentage
        self.k = k
        self.nTerms = nTerms
        self.weights = weights
        self.nDelays = nDelays
        self.single_delay_only = True if isinstance(self.nDelays, int) and self.nDelays == 1 else False

        self.maxHeight = maxHeight
        self.filename = filename
        self.problem_type = problem_type
        self.classification_metric = classification_metric
        self.nInputs = self.inputs.shape[1]
        self.nOutputs = self.outputs.shape[1]
        self.froe_mode = froe_mode
        self.pruning_probability = pruning_probability
        self.pruning_tolerance = pruning_tolerance
        self.phi_functions = phi_functions 
        self.operators = operators
        if self.evaluationMode not in ["MSE", "NMSE", "MAPE", "RMSE"]:
            raise Exception("Choose a measure between:\n" +
                            "MSE, NMSE, MAPE, or RMSE")

        if self.nInputs > 1 and self.nOutputs == 1:
            self.mode = "MISO"
        elif self.nInputs > 1 and self.nOutputs > 1:
            self.mode = "MIMO" if mode is 'NARX' else mode
            # self.mode = "FIR"
        elif self.nInputs >= 1 and self.nOutputs > 1:
            self.mode = "FIR"

        elif self.nInputs == 1 and self.nOutputs == 1:
            self.mode = "SISO"

        else:
            raise Exception("MGGP doesn't work with this system")
            

        self.element = Element(weights=self.weights,
                               nDelays=self.nDelays,
                               nInputs=self.nInputs,
                               nOutputs=self.nOutputs,
                               nTerms=self.nTerms,
                               maxHeight=self.maxHeight,
                               mode=self.mode,
                               single_delay_only=self.single_delay_only,
                               operators=self.operators)

        self.element.renameArguments(self.buildArgumentsDict())

        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", self.evaluation)

        self._mutList = []
        self._crossList = []
        self._stats = self._createStatistics()
        self._logbook = tools.Logbook()
        self._logbook.header = 'gen', 'evals', 'fitness'
        self._logbook.chapters['fitness'].header = 'min', 'avg', 'max'

        self._hofSize = int(round(self.populationSize * (self.elitePercentage / 100)))
        self._hof = tools.HallOfFame(self._hofSize)

        self._toolbox.register("select", tools.selTournament, tournsize=2)

        self.addMutation(MutGPOneTree)
        self.addMutation(MutGPUniform)
        self.addMutation(MutGPReplace)

        self.addCrossOver(CrossHighUniform)
        self.addCrossOver(CrossLowUniform)

    def addMutation(self, mutation):
        self._mutList.append(mutation(self.element))

    def addCrossOver(self, crossover):
        self._crossList.append(crossover(self.element))

    # def _delAttr(self, ind):
    #     try:
    #         del ind.fitness.values
    #         del ind.funcs
    #         del ind.kfuncs
    #         del ind.lagMax
    #     except AttributeError:
    #         pass


    def _fir_align(self, eval_type: str) -> str:
        """
        Só faz sentido para FIR.
        - OSA: usa u[k-1] -> y[k] (seu alinhamento antigo)
        - INSTANT: usa u[k] -> y[k]
        """
        if self.mode == "FIR" and eval_type == "INSTANT":
            return "INSTANT"
        return "OSA"


    def _call_with_align_if_supported(self, fn, *args, align: str):
        """
        Chama fn(*args, align=align) se o parâmetro existir.
        Caso contrário, chama fn(*args).
        """
        try:
            sig = inspect.signature(fn)
            if "align" in sig.parameters:
                return fn(*args, align=align)
        except (TypeError, ValueError):
            pass
        return fn(*args)


    def _yd_offset(self, eval_type: str) -> int:
        """
        FIR:
        - OSA => yd = y[lagMax+1:]
        - INSTANT => yd = y[lagMax:]
        """
        return 0 if (self.mode == "FIR" and eval_type == "INSTANT") else 1

    def _delAttr(self, ind):
        """Remove atributos do indivíduo de forma segura"""
        
        attrs_to_remove = ['fitness.values', '_funcs', '_lagMax', 'funcs', 'kfuncs', 'lagMax']
        
        for attr in attrs_to_remove:
            try:
                if '.' in attr:
                    parts = attr.split('.')
                    obj = ind
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            break
                    else:
                        if hasattr(obj, parts[-1]):
                            delattr(obj, parts[-1])
                else:

                    if hasattr(ind, attr):
                        delattr(ind, attr)
            except (AttributeError, TypeError):
                continue


    def stream(self):
        print(self._logbook.stream)

    def initPop(self, seed=[]):
        if len(seed) > self.populationSize: raise Exception('Seed exceeds population size!')
        if seed == []:
            self._pop = self.element._toolbox.population(self.populationSize)
        else:
            self._pop = self.element._toolbox.population(self.populationSize - len(seed))
            # self._pop += seed
            self._pop.extend([seed])
        invalid_ind = [ind for ind in self._pop if not ind.fitness.valid]

        if self.evaluationType == 'OSA':
            fitnesses = list(tqdm(self._toolbox.map(self._toolbox.evaluate, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))

        else:
            fitnesses = list(tqdm(self._toolbox.map(self.evaluation, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))
            # fitnesses = list(self._toolbox.map(self.evaluation, invalid_ind))
            # with ProcessPoolExecutor(max_workers=10) as executor:
            #     fitnesses = list(tqdm(executor.map(self.evaluation, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))

                
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = {'fitness': self._stats.compile(self._pop)}
        self._logbook.record(gen=1, evals=len(invalid_ind), **record)
        # print(self._logbook)
        self._hof.update(self._pop)


    def get_fitness_value(self, individual):
        return individual.fitness.values[0]


    def _createStatistics(self):
        stats = tools.Statistics(self.get_fitness_value)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        return stats


    def step(self, gen_number):
        if not self._pop:
            raise Exception('Population must be initialized!')

        offspring = [deepcopy(ind) for ind in self._toolbox.select(self._pop, self.populationSize - self._hofSize)]

        for i in range(0, len(offspring) - 1, 2):
            if np.random.random() < self.crossoverRate:
                cross = random.choice(self._crossList)
                offspring[i], offspring[i + 1] = cross.cross(offspring[i], offspring[i + 1])
                self._delAttr(offspring[i])
                self._delAttr(offspring[i + 1])

        for i in range(len(offspring)):
            if np.random.random() < self.mutationRate:
                mut = random.choice(self._mutList)
                offspring[i], = mut.mutate(offspring[i])
                self._delAttr(offspring[i])


        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        if self.evaluationType == 'OSA':
            fitnesses = list(tqdm(self._toolbox.map(self._toolbox.evaluate, invalid_ind), total=len(invalid_ind), desc="Evaluating Population"))

        else:
            fitnesses = list(tqdm(self._toolbox.map(self._toolbox.evaluate, invalid_ind), total=len(invalid_ind), desc="Evaluating Population"))
            # fitnesses = list(self._toolbox.map(self._toolbox.evaluate, invalid_ind))
            # with ProcessPoolExecutor(max_workers=10) as executor:
            #     fitnesses = list(tqdm(executor.map(self.evaluation, invalid_ind), total=len(invalid_ind), desc="Evaluating Population"))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # for i, ind in enumerate(offspring):
        #     if (ind.fitness.values[0] in [np.inf, -np.inf]):
        #         offspring[i] = self.element._toolbox.population(1)[0]


        self._pop = self._hof.items + offspring
        self._hof.update(self._pop)

        # self.save_model(self._hof[0])

        model = deepcopy(self._hof[0])
        self.element.compileModel(model)

        if self.mode == "FIR":
            align = self._fir_align(self.evaluationType)
            theta_value = self._call_with_align_if_supported(
                model.leastSquares, self.outputs, self.inputs, align=align
            )
        else:
            if 'sign' in self.operators: 
                theta_value = model.hysteretic_constrained_ls(self.outputs, self.inputs)
            else:
                theta_value = model.leastSquares(self.outputs, self.inputs)

        model._theta = list(theta_value)
        
        self.save_model(model)
        #---Record--Statistics-----------------------------------------------------
        record = {'fitness': self._stats.compile(self._pop)}

        self._logbook.record(gen=gen_number+1, evals=len(invalid_ind), **record)


    def buildArgumentsDict(self) -> dict:
        arguments = dict()
        
        if self.mode == "FIR":
            # Modo FIR: apenas entradas
            arguments.update({f'ARG{i}': f'u{i + 1}' for i in range(self.nInputs)})

        elif self.mode == "SISO":
            # Modo SISO: y1 e u1
            arguments.update({'ARG0': 'y1', 'ARG1': 'u1'})
        
        else:
            # Modos MISO/MIMO: saídas + entradas
            arguments.update({f'ARG{i}': f'y{i + 1}' for i in range(self.nOutputs)})
            arguments.update({f'ARG{self.nOutputs + i}': f'u{i + 1}' for i in range(self.nInputs)})
        
        return arguments


    def evaluation(self, ind: Individual) -> tuple[float]:
        try:
            self.element.compileModel(ind)

            if self.problem_type == 'regression':
                

                if self.froe_mode:
                    
                    self._constrain_phi_functions(ind)
        
                    if np.random.random() < self.pruning_probability:
                        self._apply_froe_pruning(ind)

                    theta_value = ind.hysteretic_constrained_ls(self.outputs, self.inputs)
                    ind._theta = theta_value

                    if not self._check_hysteretic_constraints(ind):
                        return (np.inf,)  
        
                else:
                    if self.mode == "FIR":
                        align = self._fir_align(self.evaluationType)
                        theta_value = self._call_with_align_if_supported(
                            ind.leastSquares, self.outputs, self.inputs, align=align
                        )
                    else:
                        theta_value = ind.leastSquares(self.outputs, self.inputs)

                    ind._theta = theta_value
                
                args = (self.outputs, self.inputs) if self.evaluationType != "MShooting" else (self.k, self.outputs, self.inputs)
                yp, yd = ind.predict(self.evaluationType, *args)
                error = ind.score(yd, yp, self.evaluationMode)
                # complexity = sum([len(subtree) for tree in ind for subtree in tree])/1000
                # fitness = error + complexity
                return error,
            
            elif self.problem_type == 'classification':
                ind._logistic_model = True
                
                if self.mode == "FIR":
                    align = self._fir_align(self.evaluationType)  # "INSTANT" ou "OSA"
                    theta_value = self._call_with_align_if_supported(
                        ind.leastSquares, self.outputs, self.inputs, align=align
                    )
                else:
                    theta_value = ind.leastSquares(self.outputs, self.inputs)
                
                ind._theta = theta_value
                
                args = (self.outputs, self.inputs) if self.evaluationType != "MShooting" else (self.k, self.outputs, self.inputs)
                
                if self.classification_metric == 'log_loss':
                    yp_proba, yd = ind.predict_proba(self.evaluationType, *args)
                    score = ind.score_classification(yd, yp_proba, 'log_loss')
                
                else:
                    yp_classes, yd = ind.predict_classes(self.evaluationType, *args)
                    score = ind.score_classification(yd, yp_classes, self.classification_metric)
                
                if self.classification_metric in ['accuracy', 'f1_macro']:
                    score = 1 - score
                
                return score,
            
            else:
                raise ValueError("problem_type must be 'regression' or 'classification'")

        except (np.linalg.LinAlgError, ValueError) as e:
            return (np.inf,)


    def run(self, seed=[]) -> None:

        print(f"System Mode: {self.mode}. N° Inputs: {self.nInputs}. N° Outputs: {self.nOutputs}")
        print(f"Input Samples: {len(self.inputs)}. Output Samples: {len(self.outputs)}\n")
        
        # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # self._toolbox.register("map", pool.map)
        self._toolbox.register("map", map)    

        init = time.time()
        self.initPop(seed=seed)
        self.stream()
        
        err_before = self._hof[0].fitness.values[0]
        err_stop_before = err_before
        counts_err = 0
        for g in range(1, self.generations):
            
            self.step(g)
            self.stream()

            # clear progressbar tqdm after 50 progressbar
            if g % 50 == 0:
                clear_output(wait=True)

            err_current = self._hof[0].fitness.values[0]
            err_stop_current = err_current

            if g%9 == 0:
                # A cada 10 gerações, verifico se melhorou em 5%, se não, aumenta a taxa de mutação em 10% até o maximo de 90%
                if (err_before/err_current -1)*100 < 5:
                    self.mutationRate = min(self.mutationRate+0.1, 0.9)

                err_before = err_current
            
            # if g%20 == 0:
                
            #     if (err_stop_before - err_stop_current) < 0.0001:
            #         # EarlyStop: Se depois de 20 épocas não teve melhoria significativa, para o treinamento
            #         break

            #     err_stop_before = err_stop_current
            
        #    if (err_stop_before - err_stop_current) < 0.0001:
        #        counts_err +=1
                
        #        if counts_err == 20:
                    # EarlyStop: Se depois de 20 épocas não teve melhoria significativa, interrompe o treinamento
        #            break
        #    else:
        #        counts_err = 0
            
        #    err_stop_before = err_stop_current


        model = self._hof[0]
        self.element.compileModel(model)

        if self.mode == "FIR":
            align = self._fir_align(self.evaluationType)
            theta_value = self._call_with_align_if_supported(
                model.leastSquares, self.outputs, self.inputs, align=align
            )
        else:
            if self.problem_type == "classification":
                model._logistic_model = True
            
            if 'sign' in self.operators: 
                theta_value = model.hysteretic_constrained_ls(self.outputs, self.inputs)
            else:
                theta_value = model.leastSquares(self.outputs, self.inputs)

        # theta_value = model.leastSquares(self.outputs, self.inputs)
        model._theta = list(theta_value)
        
        self.save_model(model)

        try:
            # print(model.to_equation())
            # print("-------------------------------------")
            print(self.simplify_model(model))
        except:
            print("----------- Model -----------")
            print(model)
            print("----------- Theta -----------")
            print(model._theta)

        self.validation_all(model=model)

        end = time.time()
        print(f"Executed in: {round(end - init, 3)} seg")


    def validation_all(self, model):
        def make_validation(dataset_validation, dataset_value=1):
            if all([value is not None for value in dataset_validation]):
                u_val, y_val = dataset_validation

                if u_val.shape[1] != self.nInputs:
                    raise Exception("The number of inputs to validate and to train must be the same")

                if y_val.shape[1] != self.nOutputs:
                    raise Exception("The number of outputs to validate and to train must be the same")

                args = (y_val, u_val) if self.evaluationTypeTest != "MShooting" else (self.k, y_val, u_val)
                
                if self.problem_type == 'regression':
                    yp, yd = model.predict(self.evaluationTypeTest, *args)
                    error = round(model.score(yd, yp, self.evaluationMode), 6)
                    print(f"{self.evaluationMode} in validation dataset {dataset_value+1}: {error}")
                
                elif self.problem_type == 'classification':
                    if self.classification_metric == 'log_loss':
                        yp_proba, yd = model.predict_proba(self.evaluationTypeTest, *args)
                        score = model.score_classification(yd, yp_proba, 'log_loss')
                        print(f"Log Loss in validation dataset {dataset_value+1}: {score:.6f}")
                    else:
                        yp_classes, yd = model.predict_classes(self.evaluationTypeTest, *args)
                        accuracy = model.score_classification(yd, yp_classes, 'accuracy')
                        f1 = model.score_classification(yd, yp_classes, 'f1_macro')
                        print(f"Accuracy in validation dataset {dataset_value+1}: {accuracy:.6f}")
                        print(f"F1-Score (macro) in validation dataset {dataset_value+1}: {f1:.6f}")

            else:
                raise Exception("Missing value of Y or U in the validation dataset")
            
        if type(self.validation) == tuple:
            make_validation(self.validation)

        elif type(self.validation) == list:
            for i, dataset_validation in enumerate(self.validation):    
                make_validation(dataset_validation, dataset_value=i)
                
        else:
            raise Exception("Choose a valuation dataset type between list or tuple.")


    def simplify_model(self, model):
        equation = model.to_equation()

        lines = equation.split('\n')
        output_lines = []
        current_output = []

        for line in lines:
            if line.startswith('Output'):
                if current_output:
                    output_lines.append(self.simplify_terms(current_output))
                    current_output = []
                output_lines.append(line)
            elif line.strip() and not line.startswith((' ', '\t')):
                current_output.append(line)

        if current_output:
            output_lines.append(self.simplify_terms(current_output))

        return '\n'.join(output_lines)


    def simplify_terms(self, terms):
      
        term_dict = {}
        for term in terms:
            if not term.strip():
                continue
            parts = term.split(' * ')
            coeff = float(parts[0].strip(' +'))
            term_part = ' * '.join(parts[1:]).strip()
            
            if term_part in term_dict:
                term_dict[term_part] += coeff
            else:
                term_dict[term_part] = coeff
        
        simplified = []
        for term_part, coeff in term_dict.items():
            simplified.append(f"{coeff:.5e} * {term_part}")
        
        return '\n'.join(simplified)
    
    
    def load_model(self, path = None):
        """
        Load a saved model to use as seed
        Args:
            filename: Name of the file containing the saved model
        Returns:
            A model instance ready to be used as seed
        """
        import pickle
        if path is None:
            with open(self.filename, 'rb') as f:
                model_data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

        element = Element(
            weights=(-1,),
            nDelays=model_data['nDelays'],
            nInputs=model_data['nInputs'],
            nOutputs=model_data['nOutputs'],
            nTerms=model_data['nTerms'],
            maxHeight=model_data['maxHeight'],
            mode=self.mode,
            # operators=model_data.get('operators', None)
        )
        element.renameArguments(model_data['arguments'])
        
        model = element.buildModelFromList(model_data['model_structure'])
        element.compileModel(model)
        model._theta = model_data['theta']
        
        return model
    

    def save_model(self, model):
        """
        Save the best model to a file for later use as seed
        Args:
            model: The model to be saved
            filename: Name of the file to save the model
        """
        import pickle
        
        model_data = {
            'model_structure': model.model2List(),
            'theta': model._theta,
            'nInputs': self.nInputs,
            'nOutputs': self.nOutputs,
            'nTerms': self.nTerms,
            'maxHeight': self.maxHeight,
            'nDelays': self.nDelays,
            'arguments': self.buildArgumentsDict(),
            'operators': self.operators 
        }
        
        with open(self.filename, 'wb') as f:
            pickle.dump(model_data, f)


    def _check_hysteretic_constraints(self, ind, tol=1e-10):
        """
        Verifica se o modelo atende às restrições de continuum de equilíbrio
        """
        try:
            clusters = ind.identify_term_clusters(self.outputs, self.inputs)
            theta = np.asarray(ind._theta).flatten() 
            
            linear_output_sum = sum(theta[idx] for idx in clusters['linear_output'])
            if abs(linear_output_sum - 1.0) > tol:
                return False
                
            for cluster_type in ['linear_input', 'cross_terms', 'nonlinear_y', 'nonlinear_u']:
                cluster_sum = sum(theta[idx] for idx in clusters[cluster_type])
                if abs(cluster_sum) > tol:
                    return False
                    
            return True
        except:
            return False
        

    def get_terminal_by_name(self, name):

        if isinstance(self.element._pset.terminals, defaultdict):

            for term in self.element._pset.terminals[object]:
                if hasattr(term, 'value') and term.value == name:
                    return term
        else:
            for term in self.element._pset.terminals:
                if hasattr(term, 'value') and term.value == name:
                    return term
        return None
    

    def _constrain_phi_functions(self, ind):
        """
        Garante que as funções φ (subtraction e sign) recebam apenas 
        variáveis de entrada defasadas.
        - Primeira substituição dentro de φ: u1
        - Substituições subsequentes dentro da mesma φ: q1(u1)
        """
        
        PHI_NAMES = {"subtraction", "sign"}

        def _is_q_chain_to_u(tree: gp.PrimitiveTree, idx: int) -> bool:
            """True se a subárvore em idx é q*(u#) (uma cadeia de q's terminando em u)."""
            node = tree[idx]

            if isinstance(node, gp.Terminal):
                return isinstance(node.value, str) and node.value.startswith("u")

            if not isinstance(node, gp.Primitive):
                return False
            if not node.name.startswith("q"):
                return False
            if node.arity != 1:
                return False

            child_idx = idx + 1
            return _is_q_chain_to_u(tree, child_idx)


        def _make_q1_u1_tree(pset) -> gp.PrimitiveTree:
            q1 = next(p for p in pset.primitives[pset.ret] if p.name == "q1")
            u1 = next(t for t in pset.terminals[pset.ret] if getattr(t, "value", None) == "u1")
            return gp.PrimitiveTree([q1, u1])
        
        
        def _make_u1_tree(pset) -> gp.PrimitiveTree:
            u1 = next(t for t in pset.terminals[pset.ret] if getattr(t, "value", None) == "u1")
            return gp.PrimitiveTree([u1])


        def constrain_phi_tree(tree: gp.PrimitiveTree, pset) -> gp.PrimitiveTree:
            """Garante que cada argumento de φ é q*(u#). Se não for, vira q1(u1)."""
            i = 0
            repl = _make_q1_u1_tree(pset)
            rep_u1 = _make_u1_tree(pset)

            while i < len(tree):
                node = tree[i]
                if isinstance(node, gp.Primitive) and node.name in PHI_NAMES:
                    
                    arg_idx = i + 1
                    first_substitution = True
                    for _ in range(node.arity):
                        arg_slice = tree.searchSubtree(arg_idx)
                        if not _is_q_chain_to_u(tree, arg_idx):
                            if first_substitution:
                                tree[arg_slice] = rep_u1
                                first_substitution = False

                                arg_idx = arg_slice.start + len(rep_u1)
                            else:
                                tree[arg_slice] = repl
                                
                                arg_idx = arg_slice.start + len(repl)
                        else:
                            arg_idx = arg_slice.stop
                i += 1
            return tree

        if self.mode in ["SISO", "MISO"] or (self.mode == "FIR" and self.nOutputs == 1):
            for i, tree in enumerate(ind):
                ind[i] = constrain_phi_tree(tree, self.element._pset)

        else:  # MIMO
            for o in range(len(ind)):
                for i, tree in enumerate(ind[o]):
                    ind[o][i] = constrain_phi_tree(tree, self.element._pset)


    def _is_lagged_input(self, node):
        """Verifica se o nó é uma variável de entrada defasada"""
        if isinstance(node, gp.Terminal) and node.value.startswith('u'):
            return True
        return False


    def _apply_froe_pruning(self, ind):
        """
        Aplica o algoritmo FROE para remover termos com ERR baixo
        """

        if self.mode == "FIR":
            align = self._fir_align(self.evaluationType)

            P = self._call_with_align_if_supported(
                ind.makeRegressors, self.outputs, self.inputs, align=align
            )

            yd = self.outputs[ind.lagMax + self._yd_offset(self.evaluationType):]

        else:
            P = ind.makeRegressors(self.outputs, self.inputs)
            yd = self.outputs[ind.lagMax + 1:]
        
        if self.mode in ["SISO", "MISO"] or (self.mode == "FIR" and self.nOutputs == 1):
            self._froe_pruning_miso(ind, P, yd)
        else:  # MIMO
            self._froe_pruning_mimo(ind, P, yd)


    def _froe_pruning_miso(self, ind, P, yd):
        """FROE para modelos MISO"""
        n_terms = P.shape[1] - 1 
        
        err_values = []
        for j in range(1, n_terms + 1): 
            w_j = P[:, j] 
            g_j = np.dot(w_j, yd) / np.dot(w_j, w_j)  
            err_j = (g_j**2 * np.sum(w_j**2)) / np.sum(yd**2)  # ERR
            err_values.append((j, err_j))
        
        err_values.sort(key=lambda x: x[1], reverse=True)
        terms_to_keep = [0] 
        
        for j, err in err_values:
            if err >= self.pruning_tolerance:  
                terms_to_keep.append(j)
            else:
                if (j-1) < len(ind):
                    ind[j-1] = self.element._toolbox._program() 
        
        new_ind = []
        for i in range(len(ind)):
            if i in [x-1 for x in terms_to_keep if x > 0]:
                new_ind.append(ind[i])
        
        ind[:] = new_ind


    def _froe_pruning_mimo(self, ind, P, yd):
        """FROE para modelos MIMO (aplica para cada saída)"""
        for o in range(len(ind)):
            P_o = P[o]  
            yd_o = yd[:, o] if yd.ndim > 1 else yd
            
            n_terms = P_o.shape[1] - 1
            err_values = []
            
            for j in range(1, n_terms + 1):
                w_j = P_o[:, j]
                g_j = np.dot(w_j, yd_o) / np.dot(w_j, w_j)
                err_j = (g_j**2 * np.sum(w_j**2)) / np.sum(yd_o**2)
                err_values.append((j, err_j))
            
            err_values.sort(key=lambda x: x[1], reverse=True)
            terms_to_keep = [0]
            
            for j, err in err_values:
                if err >= self.pruning_tolerance:
                    terms_to_keep.append(j)
                else:
                    if (j-1) < len(ind[o]):
                        ind[o][j-1] = self.element._toolbox._program()
            
            new_output = []
            for i in range(len(ind[o])):
                if i in [x-1 for x in terms_to_keep if x > 0]:
                    new_output.append(ind[o][i])
            
            ind[o][:] = new_output