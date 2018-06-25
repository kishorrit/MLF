# Chapter 9 Fighting Bias

https://fairmlclass.github.io/

We like to think that machines are more rational than us. Heartless silicon applying cold logic to optimize some outcome. Thus, when automated decision making entered the economy, many hoped that computers would reduce prejudice and discrimination. But computers are made and trained by humans. Their data stems from an unjust world. And if we are not careful, they will amplify our biases.

In the financial industry, anti-discrimination is not only a matter of moral. Take for instance the Equal Credit Opportunity Act (ECOA) which came into force in 1974. It explicitly forbids creditors to discriminate applicants based on race, sex, marital status and a number of other attributes. It also requires creditors to inform applicants about the reasons for denial. 

The algorithms discussed in this books are discrimination machines. They will find the features on which best to discriminate to achieve a given objective. However, discrimination is **domain specific**. It is okay to target ads for books from a certain country to people who are also from that country. It is not okay to deny a loan to people from a certain country. In the financial domain, there are much stricter rules for discrimination than in book sales, because decisions in the financial domain have a much more severe impact on peoples lives. Equally, discrimination is **feature specific**. It is okay to discriminate loan applicants on their history of repaying loans, but not on their country of origin.

Equally, the algorithms discussed in this book are feature extraction algorithms. Even if regulated features are omitted, they might infer them from proxy features and then discriminate based on them anyway. Zip codes for instance can be used to predict race reasonably well in many cities in the United States. Omitting regulated features is not enough.

This chapter discusses where bias in machine comes from, its legal implications, and how it can be reduced.

# Sources of unfairness in ML

As discussed many times in this book, models are a function of the data they are trained on. Generally, more data leads to smaller errors. By definition, there is less data on minority groups, simply because there are fewer people in the group. This **disparate sample size** can lead to worse model performance for the minority group. This increased error is often a **systematic error**. The model might have overfit to majority group data, so that the relationships it found do not hold on the minority group data. Since there is little minority group data, this is not punished as much. Imagine you are training a credit scoring model, and the vast majority of your data comes from people living in lower manhattan and a small minority lives in rural areas. Manhattan housing is much more expensive, so the model might learn that you need a very high income to buy an apartment. Rural housing is much cheaper, but because the model is largely trained on manhattan data, it might deny loan applications to rural applicants because they also tend to have lower incomes than their manhattan peers.

Next to sample size issues, our data can be biased by itself. 'Raw Data' does not exist. Data does not appear naturally but is measured by humans using human made measurement protocols. These protocols can be biased in many ways. They can have **sampling biases**, like in the manhattan housing example. They can have **measurement biases**. Your measurement might not measure what it is intended to measure or discriminate against one group. One example are Eurocentric knowledge tests that ask about the tales of the brothers Grimm, but not about Indian fairy tales. And finally, there can be **pre-existing social biases**. These are visible in word vectors for instance. In Word2Vec, the vector mapping from father to doctor in latent space maps from mother to nurse. The vector from man to computer programmer maps from woman to homemaker. This is because sexisim is encoded in the written language of a sexist society. Until today, doctors are usually men and nurses are usually women. Tech-companies diversity statistics reveal that far more men are computer programmers. These biases get encoded in models, 

Intro
http://mrtz.org/nips17/#/11

# Legal perspectives
There are two doctrines in anti discrimination law, disparate treatment and disparate impact. **Disparate treatment** can be formal, that is if regulated features are explicitly used for discrimination, which is obviously not legal. But it can also be a problem if it is not formal but intentional. Intentionally discriminating against zip codes with the hope of discriminating against race is also not legal. Disparate treatment problems have less to do with the algorithm and more with the organization running it. **Disparate impact** can be a problem if an algorithm is deployed that has a different impact on different groups, even without the organization knowing about it. Let's walk through a lending scenario in which disparate impact could be a problem: First, the plaintiff must establish that there is a disparate impact. This is usually done with the **four fifths rule**: If the selection rate of a group is less then 80% of the group with the highest selection rate of, it is regarded as evidence of adverse impact. If a lender has 150 loan applicants from group A, of which 100, or 67% are accepted and 50 applicants from group B of which 25 are accepted, the difference in selection is 0.5/0.67 = 0.746, which qualifies as evidence for discrimination against group B. To this, the defendant can counter by showing that the decision procedure is justified as a necessity. Finally, the the plaintiff has the opportunity to show that the goal of the procedure could also be achieved with a different procedure that shows a smaller disparity.

The disparate treatment doctrine tries to achieve procedural fairness and equal opportunity. The disparate impact doctrine aims for distributive justice and minimized inequality in outcomes. There is an intrinsic tension between the two doctrines, illustrated by the *Ricci v. DeStefano* case from 2009. In this case, nineteen white and one Hispanic firefighters sued their employer, the New Haven Fire Department. The firefighters had all passed their test for promotion. Yet their black colleagues did not score high enough for promotion. Fearing an disparate impact lawsuit, the city invalidated the test results and did not promote the firefighters. Because the evidence for disparate impact was not strong enough, the supreme court eventually ruled that the firefighters should have been promoted.

Given the complex legal and technical situation around fairness in machine learning, we will next dive into how we can define and quantify fairness, before using this insight to create more fair models.

# Observational fairness

Thresholds and equal opportunity
https://research.google.com/bigpicture/attacking-discrimination-in-ml/

Equality is often seen as a purely qualitative issue, and as such, often dismissed buy quantitative minded modelers. As this section shows, equality can be seen from a quantitative perspective, too. 
Consider a classifier $c$ with input $X$, some sensitive input $A$, a target $Y$ and output C. Usually we note the classifier output as $\hat{Y}$, but for readability we follow CS 294 and name it $C$.

Say our classifier is used to decide who gets a loan. When would we consider this classifier to be fair? In order to answer this question, picture two demographics, Group A and B, of loan applicants. Given a credit score, our classifier has to find a cutoff point. Below you can see the distribution of applicants. In orange are applicants who would not have repaid the loan and did not get accepted, true negatives (TN). In blue are applicants who would have repaid the loan but did not get accepted, false negatives (FN). In yellow are applicants who did get the loan but did not pay it back, false positives (FN). In grey are applicants who did receive the loan and paid it back, true positives (TP). The data for this example is synthetic, you can find the excel file used for these calculations in the GitHub repository of this book.

For this exercise we assume that a successful applicant and pays yields a profit of \$300 while a defaulting successful applicant costs \$700. The cutoff point below has been chosen to maximize profits:

![Max Profit](./assets/max_profit.png)

As you can see, there are several issues with this choice of cutoff point. group B applicants need to have a better score to get a loan than group A applicants, indicating disparate treatment. At the same time, about 51% of group A applicants get a loan but only 37% of group B applicants, indicating disparate impact. A **group unaware threshold** would give both groups the same minimum score:

![Equal Cutoff](./assets/equal_cutoff.png)

Both groups have the same cutoff rate, but group A has been given fewer loans. At the same time, predictions for group A have a lower accuracy than for group B. It seems, that although both groups face the same score threshold, group A is at a disadvantage.

**Demographic parity** aims to achieve fairness by ensuring that both groups have the same chance of receiving the loan. This method aims to achieve the same selection rate for both groups, which is what impact disparity is measured by. Mathematically this can be expressed as:

$$P(C=1∣A=1)=P(C=1∣A=0)$$

If we apply this rule to our data, we arrive at the following cutoff points:
![Equal Pick Rate](./assets/equal_pick_rate.png)

While this method can not be blamed for statistical discrimination and disparate impact, it can be blamed for disparate treatment. Group A is given a lower threshold score and more successful group A applicants default on their loans. In fact, group A is not profitable and gets subsidized by group B. Accepting a worse economic outcome to favor a certain group is also called taste based discrimination. It could be said that the higher thresholds for group B are unfair, as they have a lower false positive rate.

**True positive parity**, also called equal opportunity, means that both demographics have the same true positive rate. For people who can pay back the loan, the same chance of getting a loan should exist. Mathematically:
$$P(C=1∣Y=1,A=1)=P(C=1∣Y=1,A=0)$$

Applied to our data, this policy looks similar to demographic parity, except that the group A cutoff point is even lower:
![Equal opportunity](./assets/equal_opportunity.png)

Equal opportunity can address many of the problems of demographic parity. Most people believe that everyone should be given the same opportunity. Still, our classifier is less accurate for group A and there is a form of disparate treatment in place.

**Accuracy parity** prescribes that the accuracy of predictions should be the same for both groups. Mathematically this can be expressed as:

$$P(C=Y∣A=1)=P(C=Y∣A=0)$$

The probability that the classifier is correct should be the same for the two possible values of the sensitive variable $A$. When we apply this criteria to our data, we arrive at the following output:

![Equal Accuracy](./assets/equal_acc.png)

The downside becomes apparent from the graphic above. In order to satisfy the accuracy constraint, members of Group B are given much easier access to loans. 

Tradeoffs are necessary, no classifier can have precision parity, true positive parity _and_ false positive parity, unless the classifier is perfect, $C=Y$ or both demographics have the same base rates:
$$P(Y=1|A=1)=P(Y=1|A=0)$$

There are many more ways to express fairness in different ways. The key takeaway however is that none of them perfectly satisfies all fairness criteria. For any two populations with unequal base rates, unequal chances of repaying their loan, establishing statistical parity requires introducing a treatment disparity. This fact has led to much debate and the best practice to express and eliminate discrimination has not been agreed on, yet. Yet, even if the perfect mathematical expression of fairness was found, it would not immediately lead to perfectly fair systems. Any machine learning algorithm is part of a bigger system. Inputs $X$ are often not clearly defined as different algorithm in the same system might use different inputs. Demographic groups $A$ are often not clearly defined or inferred. Even the output $C$ of the classifier can often not be clearly distinguished, as many algorithms together might perform the classification task while each algorithm is predicting a different output, like a credit score and a profitability estimate. **Good technology is not a substitute for good policy**. Blindly following an algorithm without an opportunity for individual consideration or appeal will always lead to unfairness. But while mathematical fairness criteria can not solve all fairness issues, it is surely worth trying to make machine learning algorithms more fair. This is what the next section is about.

# Training to be fair 
Training models to be fair
https://blog.godatadriven.com/fairness-in-ml

Data 
https://archive.ics.uci.edu/ml/datasets/Adult

There are multiple ways to train models to be more fair. A simple approach could be to use the different fairness measures we have listed above as an additional loss. In practice, this approach has turned out to not work very well. The models produced with such 'fairness regularization' turn out to have poor performance on the actual classification task. An alternative approach is to use and adversarial network. In 2016, Louppe, Kagan and Cranmer published 'Learning to Pivot with Adversarial Networks', a paper which shows how to use an adversarial network to train a classifier to ignore a nuisance parameter, such as a sensitive feature.

In this example, we will train a classifier to predict if an adult makes over \$50K in annual income. The challenge is to make our classifier unbiased from influences of race and gender, and only focus on features we can discriminate on such as their occupation and gains they make from capital. To this end we train a classifier and an adversarial network. The adversarial aims to classify the sensitive attributes $a$, gender and race, from the predictions of the classifier.

![pivot](./assets/pivot.png)

The classifier aims to classify by income but also aims to fool the adversarial. The classifiers minimization objective is:

$$\min[L_y - \lambda L_A]$$

Where $L_y$ is a binary cross-entropy loss of the classification and $L_A$ is the adversarial's loss. $\lambda$ is a hyper parameter we can use to amplify or reduce the impact of the adversarial loss.

This implementation of the adverserial fairness method follows an implementation by Stijn Tonk and Henk Griffioen. You can find the code to this chapter on Kaggle under: https://www.kaggle.com/jannesklaas/learning-how-to-be-fair

To train fair, we need not only data X and targets y, but also data about sensitive attributes, A. We take data from the 1994 US census provided by the UCI repository https://archive.ics.uci.edu/ml/datasets/Adult

To make loading the data easier, it has been transformed into a CSV with column headers. Please refer to the online version to see the data as well.

First we load the data. The dataset contains data about people from many races, but for simplicity we will only focus on white and black people for the race attribute.
```Python 
path = '../input/adult.csv'
input_data = pd.read_csv(path, na_values="?")
input_data = input_data[input_data['race'].isin(['White', 'Black'])]
```

We select the sensitive attributes race and gender into our sensitive dataset A. We one hot encode the data so that 'Male' equals one for the gender attribute and 'White' equals one for the race attribute.
```Python 
sensitive_attribs = ['race', 'gender']
A = input_data[sensitive_attribs]
A = pd.get_dummies(A,drop_first=True)
A.columns = sensitive_attribs
```

Our target is the income attribute. We encode '>50K' as 1 and everything else as zero.
```python 
y = (input_data['income'] == '>50K').astype(int)
```

To get our training data, we first remove the sensitive and target attributes. We then fill all missing values and one hot encode all data.
```Python 
X = input_data.drop(labels=['income', 'race', 'gender'],axis=1)

X = X.fillna('Unknown')

X = pd.get_dummies(X, drop_first=True)
```

Finally, we split the data into train and test split. We stratify the data to ensure that the same amount of high earners are in test and training data.
```Python 
X_train, X_test, y_train, y_test, A_train, A_test = \
train_test_split(X, y, A, test_size=0.5, 
                stratify=y, random_state=7)
```

To ensure the data works nicely with the neural network, we scale the data using scikit learn's `StandardScaler`:
```Python 
scaler = StandardScaler().fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), 
                       columns=X_train.columns, 
                       index=X_train.index)
                       
X_test = pd.DataFrame(scaler.transform(X_test), 
                      columns=X_test.columns, 
                      index=X_test.index)
```

We need a metric how fair our model is. We are using the disparate impact selection rule. The `p_rule` method calculates the share of people classified to have over \$50K income from both groups and returns the ratio of selections in the disadvantaged demographic over the ratio of selections in the advantaged group. The goal is for the `p_rule` method to return at least 80% the meet the four fifths rule for both race and gender. This function is only used for monitoring, and not as a loss function. 

```Python 
def p_rule(y_pred, a_values, threshold=0.5):
    y_a_1 = y_pred[a_values == 1] > threshold if threshold else y_pred[a_values == 1] #1
    y_a_0 = y_pred[a_values == 0] > threshold if threshold else y_pred[a_values == 0] 
    odds = y_a_1.mean() / y_a_0.mean() #2
    return np.min([odds, 1/odds]) * 100 
```
\#1 First, select who is selected given a selection threshold. Here, we classify everyone whom the model assigns a chance of over 50% to make \$50K as a high earner. 

\#2 Then we calculate the selection ratio of both demographics. We divide the ratio of the one group by the ratio of the other group. By returning the minimum of either the odds or one divided by the odds, we ensure to returns a value below 1.


```Python 
n_features=X_train.shape[1]
n_sensitive=A_train.shape[1]
lambdas=[130., 30.]
```

```Python 
clf_inputs = Input(shape=(n_features,)) # Classifier input = All features

############### Create CLF net ########################
x = Dense(32, activation='relu')(clf_inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid', name='y')(x)
clf_net = Model(inputs=[clf_inputs], outputs=[outputs])
#######################################################
```

```Python 
adv_inputs = Input(shape=(1,)) # Adversary input = Classifier output (one number)

############## Create ADV net #########################
x = Dense(32, activation='relu')(adv_inputs)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = [Dense(1, activation='sigmoid')(x) for _ in range(n_sensitive)]
adv_net = Model(inputs=[adv_inputs], outputs=outputs)
#######################################################
```

```Python 
def make_trainable_fn(net): # Produces a function that makes a network trainable or not
    def make_trainable(flag): # Loop over layers and set their trainability
        net.trainable = flag
        for layer in net.layers:
            layer.trainable = flag
    return make_trainable
```

```Python 
############## Create train switches #################
trainable_clf_net = make_trainable_fn(clf_net) # Get function to make classifier trainable

trainable_adv_net = make_trainable_fn(adv_net) # Function to make adversary trainable

######################################################
```

```Python 
clf = clf_net
trainable_clf_net(True)
clf.compile(loss='binary_crossentropy', optimizer='adam')

```

```Python 
# Creates a classifier adversary super net
clf_w_adv = Model(inputs=[clf_inputs], 
                  outputs=[clf_net(clf_inputs)]+adv_net(clf_net(clf_inputs)))

# The adversary is not trainable the classifier is
trainable_clf_net(True)
trainable_adv_net(False)
# Create a weighted loss for all sensitive variables
loss_weights = [1.]+[-lambda_param for lambda_param in lambdas]
# Compile super net
clf_w_adv.compile(loss=['binary_crossentropy']*(len(loss_weights)), 
                  loss_weights=loss_weights,
                  optimizer='adam')
```

```Python 
# Compile adversary with the classifier as inputs
adv = Model(inputs=[clf_inputs], outputs=adv_net(clf_net(clf_inputs)))
# Classifier is not trainable, adversary is
trainable_clf_net(False)
trainable_adv_net(True)
adv.compile(loss=['binary_crossentropy']*n_sensitive, optimizer='adam')
```

```Python 
trainable_clf_net(True)
clf.fit(X_train.values, y_train.values, epochs=10)
```

```Python 
trainable_clf_net(False)
trainable_adv_net(True)
class_weight_adv = compute_class_weights(A_train)
adv.fit(X_train.values, 
        np.hsplit(A_train.values, A_train.shape[1]), 
        class_weight=class_weight_adv,epochs=10)
```

```Python 
y_pred = clf.predict(X_test)
```

```Python 
for sens in A_test.columns:
    pr = p_rule(y_pred,A_test[sens])
    print('{}: {:.2f}%'.format(sens,pr))
```

```
race: 41.71%
gender: 29.41%
```

```Python
def fit(x, y, a, validation_data, n_iter=250, batch_size=128):
    n_sensitive = a.shape[1]
    
    x_val, y_val, a_val = validation_data

    class_weight_adv = compute_class_weights(a)
    
    class_weight_clf_w_adv = [{0:1., 1:1.}]+class_weight_adv
    
    val_metrics = pd.DataFrame()
    
    fairness_metrics = pd.DataFrame()
    
    for idx in range(n_iter): # Train for n epochs
        
        # train adverserial
        trainable_clf_net(False)
        trainable_adv_net(True)
        adv.fit(x.values, np.hsplit(a.values, a.shape[1]), batch_size=batch_size, 
                      class_weight=class_weight_adv, epochs=1, verbose=0)
        
        
        # train classifier
        # Make classifier trainable and adversery untrainable
        trainable_clf_net(True)
        trainable_adv_net(False)
        # Sample batch
        indices = np.random.permutation(len(x))[:batch_size]
        # Train on batch
        clf_w_adv.train_on_batch(x.values[indices], 
                                [y.values[indices]]+np.hsplit(a.values[indices], n_sensitive),
                                class_weight=class_weight_clf_w_adv)
        
        if validation_data is not None:
            # Make validation data predictions
            y_pred = pd.Series(clf.predict(x_val).ravel(), index=y_val.index)
            
            roc_auc = roc_auc_score(y_val, y_pred)
            acc = accuracy_score(y_val, (y_pred>0.5))*100
            # Calculate ROC and accuracy
            val_metrics.loc[idx, 'ROC AUC'] = roc_auc
            val_metrics.loc[idx, 'Accuracy'] = acc
            
            # Calculate p rule
            for sensitive_attr in a_val.columns:
                fairness_metrics.loc[idx, sensitive_attr] = p_rule(y_pred,a_val[sensitive_attr])
                
            print('Epoch: {}, Accuracy: {:.2f}, Race P: {:.2f}, Gender P: {:.2f}'.format(idx,
                                                                                         acc, 
                                                                                         fairness_metrics.loc[idx, 'race'],
                                                                                         fairness_metrics.loc[idx, 'gender']))
            
    return val_metrics,fairness_metrics
```

```Python 
# adverserial train on train set and validate on test set
vm, fm = fit(X_train, y_train, A_train,
             validation_data=(X_test, y_test, A_test),n_iter=165)
```

# Beyond observational fairness 

- Interpretability
https://geomblog.github.io/fairness/
- Causal ML
