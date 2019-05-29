# The crazy joblib version that does not seem to work, at least not in a notebook

def unwrap_self_fit(s,f,i):
    print('Chunk ' + str(i))
    return CompositeModel.fit_single_chunk(s,f,i)

class CompositeModel:    
    
    def __init__(self, split_column, target_column):
        self._split_column = split_column
        self._target_column = target_column
        self._item_run_map = dict()
        self._item_model_map = dict()
        self._model_impls = dict()
        self._indices = dict()
    

    # todo: redo this so that this class splits its own training data
    # on the splitcolumn, ideally in sequence, rather than materializing
    # the whole dataset again in memory as a sequence of chunks. 
    # Just decide on a number of jobs, say 4, and then make 4 chunks, 
    # run them in parallel, discard them, and go again.
    # Use the accumulator pattern from joblib docs.
    def fit(self, train_frames, indices):
    
        from joblib import Parallel, delayed
    
        self._indices = indices
            
        # Warning: you probably don't want n_jobs too big - 
        # the reason for this whole exercise is that memory is limited.
        # But helps get through testing faster
        Parallel(n_jobs=2, require='sharedmem')
        (
            delayed(unwrap_self_fit)((self, train_frames[idx], idx)) 
            for idx in range(2) # range(len(train_frames))
        )
        return True
        
    def fit_single_chunk(self, Xy, idx):
            
        # TODO: this is hard-wired and overwrites outer scope        
        time_series_settings = {
            'time_column_name': time_colname,
            'grain_column_names': grain_colnames,
            'drop_column_names': [],
            'max_horizon': 31
        }
                        
        X_train = Xy.copy()            
        y_train = X_train.pop(self._target_column).values                        
          
        automl_config = AutoMLConfig(task='forecasting',
                            debug_log='automl-grocery.log',
                            primary_metric='normalized_root_mean_squared_error',
                            iterations=10,
                            X=X_train,
                            y=y_train,                             
                            n_cross_validations=3,
                            enable_ensembling=False,
                            path=project_folder,
                            verbosity=logging.INFO,    
                            **time_series_settings)
        
        # get the model and metadata
        local_run = experiment.submit(automl_config, show_output=True) # Parent run 
        best_run, fitted_pipeline = local_run.get_output()             # Favorite child
        model_id = local_run.model_id
        
        # record the model for item
        self._model_impls[model_id] = fitted_pipeline
        for idx, item in enumerate(self._indices[idx]):
            self._item_model_map[item] = model_id
            self._item_run_map[item] = best_run.id
            
        return True
                
    def forecast(self, X_test, y_test):
        
        # split X and y together by splitcolumn
        X_copy = X_test.copy()
        X_copy['__automl_target_column'] = y_test
        chunks = split_into_chunks_by_groups(X_copy, self._split_column, self._indices)
        
        ys = []
        X_transes = []
        for chunk in chunks:
            # Look up the right model. It should be the same model 
            # for the whole chunk by construction
            item = chunk.loc[X_copy.index[0], self._split_column]
            modelid = self._item_model_map[item]
            model = self._model_impls[modelid]
            
            y_chunk = chunk.pop(self._target_column)
            y_pred, X_trans = model.forecast(chunk, y_chunk)
            ys.append(y_chunk)
            X_transes.append(X_trans)
            
        return pd.concat(ys), pd.concat(X_transes)                    