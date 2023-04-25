statistic_terms = {
    "waged average":"Σ(w_i * x_i) / Σw_i\n\t Multiply each value by its corresponding weight, sum the products, and then divide the total by the sum of the weights",
    "variance":"Σ[(X_i - X_mean)²/(n - 1)]\n\t Sum of squared differences between set's elements and its mean divide by its length",
    "standard deviation":"Square root from sum of squared differences between set's elements and its mean divide by its length",
    "mad":"Σ[abs(X_i - X_mean)/(n - 1)]\n\t Average of differences between each set's elements and its mean divided by all sets lenghts sum\n\t Not recomended for more than 2 samples",
    "covariance":"Σ[(X_i - X_mean)*(Y_i - Y_mean)]/(n - 1)\n\t The sum of products of all sets differences. Degree to which two variables are linearly related\n\t Only for 2 samples",
    "pooled variance":"Σ[(n_i - 1)*var_i] / Σ[n]- 2\n\t Only for two samples",
    "type1 error": "Rejecting a true null hypothesis (false positive)",
    "type2 error": "Failing to reject a false null hypothesis (false negative)",
    "tvalue": "(mean1 - mean2) / sqrt( pldvar/n1 + pldvar/n2 )\n\t Standardized difference between the means of two samples; t>2.5 significance, t<0.05 no significant difference",
    "pvalue":"Probability of observing a difference in means as large or larger than the one we observed in our sample\n\t To find in t-distribution table ",
    "clt": "Central limit theorem - a statistical theory that states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases",

    }

while True:
    term = input("Term: ")
    if term not in statistic_terms.keys():
        term
    else:
        print("\t",term,"-",statistic_terms[term])
