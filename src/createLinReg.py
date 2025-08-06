def createLinReg(x_train, x_test, y_train, print_vals = False):
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    # train linear model
    try:
        lrmodel = model.fit(x_train, y_train)
    except Exception as e:
            print(f"Error occurred: {e}")
            return 0
    # Make predictions on train and test sets
    lr_train_pred = lrmodel.predict(x_train)
    lr_test_pred = lrmodel.predict(x_test)
    
    if print_vals:
        print("Model Coefficients:")
        print(lrmodel.coef_)
        print('\n')
        print("Model Intercept:")
        print(lrmodel.intercept_)
        print('\n')
        print("Number of Coefficients:")
        print(len(lrmodel.coef_))
    return lrmodel, lr_train_pred, lr_test_pred
