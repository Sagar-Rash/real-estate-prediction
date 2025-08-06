

def createDTree(x_train, y_train, x_test, features, max_depth=3, max_features=10,random_state=567, plot_tree = False):
    # Create decision tree with set parameters given. max features = 10 by default
    from sklearn.tree import DecisionTreeRegressor

    try:
        dt = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state)
        # Train    
        dtmodel = dt.fit(x_train, y_train)
        # Training Predictions
        dt_train_pred = dtmodel.predict(x_train)
    
        # Test Predictions
        dt_test_pred = dtmodel.predict(x_test)
    except Exception as e:
            print(f"Error occurred: {e}")
            return -1
    # If the user wants to plot the decision tree, plot the tree
    if plot_tree:
        from matplotlib.pyplot import show, savefig
        from sklearn import tree
        # Plot the tree with feature names
        print("Tree Plot")
        try:
            features = features[0:max_features]
            tree.plot_tree(dtmodel, feature_names = features)
            # Save Figure
        
            savefig(r'photos\CST2216_DTree.png', dpi=1000)
        
            show()
        except Exception as e:
            print(f"Error occurred: {e}")
            return 0
    
    return dtmodel, dt_train_pred, dt_test_pred


