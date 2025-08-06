

def createRdmFrst(x_train, x_test, y_train, n_est = 200, max_d = 2, max_feat = 10, crit='absolute_error', feat_imp = False):
    # Create random forest regressor with given inputs and hyperparameters
    # Returns RF model, train predictions, and test predictions
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = n_est, max_depth = max_d, max_features = max_feat, criterion = crit)
    try:
        # Train the model
        rfmodel = rf.fit(x_train,y_train)
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0
    # Make predictions on train and test data
    rf_train_pred = rfmodel.predict(x_train)
    rf_test_pred = rfmodel.predict(x_test)
    
    if feat_imp:
        # Print feature importances if prompted
        print("Feature Importances:")
        print(rfmodel.feature_importances_)

    return rfmodel, rf_train_pred, rf_test_pred