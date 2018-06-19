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

Training models to be fair
https://blog.godatadriven.com/fairness-in-ml

Data 
https://archive.ics.uci.edu/ml/datasets/Adult

Equality is often seen as a purely qualitative issue, and as such, often dismissed buy quantitative minded modelers. As this section shows, equality can be seen from a quantitative perspective, too. 
Consider a classifier $c$ with input $X$, some sensitive input $A$, a target $Y$ and output C. Usually we note the classifier output as $\hat{Y}$, but for readability we follow CS 294 and name it $C$.

Say our classifier is used to decide who gets a loan. When would we consider this classifier to be fair? In order to answer this question, picture two demographics, Group A and B, of loan applicants. Given a credit score, our classifier has to find a cutoff point. Below you can see the distribution of applicants. In orange are applicants who would not have repaid the loan and did not get accepted, true negatives (TN). In blue are applicants who would have repaid the loan but did not get accepted, false negatives (FN). In yellow are applicants who did get the loan but did not pay it back, false positives (FN). In grey are applicants who did receive the loan and paid it back, true positives (TP). The data for this example is synthetic, you can find the excel file used for these calculations in the GitHub repository of this book.

For this exercise we assume that a successful applicant and pays yields a profit of \$300 while a defaulting successful applicant costs \$700. The cutoff point below has been chosen to maximize profits:

![Max Profit](./assets/max_profit.png)

As you can see, there are several issues with this choice of cutoff point. group B applicants need to have a better score to get a loan than group A applicants, indicating disparate treatment. At the same time, about 51% of group A applicants get a loan but only 37% of group B applicants, indicating disparate impact. A **group unaware threshold** would give both groups the same minimum score:

![Equal Cutoff](./assets/equal_cutoff.png)

Both groups have the same cutoff rate, but group A has been given fewer loans. At the same time, predictions for group A have a lower accuracy than for group B. It seems, that although both groups face the same score threshold, group A is at a disadvantage.

**Accuracy parity** prescribes that the accuracy of predictions should be the same for both groups. Mathematically this can be expressed as:

$$P(C=Y∣A=1)=P(C=Y∣A=0)$$

The probability that the classifier is correct should be the same for the two possible values of the sensitive variable $A$. When we apply this criteria to our data, we arrive at the following output:

![Max Profit Equal Cutoff](./assets/equal_acc.png)

The downside becomes apparent from the graphic above. In order to satisfy the accuracy constraint, members of Group B are given much easier access to loans. 


**Demographic parity** aims to achieve fairness by ensuring that both groups have the same chance of recieving the loan.


$$P(C=1∣A=1)=P(C=1∣A=0)$$

Precision parity, both demographics have the same precision rate
$$P(Y=1∣C=1,A=1)=P(Y=1∣C=1,A=0)$$

$$Precision = \frac{TP}{TP + FP}$$

True positive parity, both demographics have the same true positive rate 
$$P(C=1∣Y=1,A=1)=P(C=1∣Y=1,A=0)$$

Tradeoffs are nessecary, no classifier can have precision parity, true positive parity _and_ false positive parity, unless the classifier is perfect, $C=Y$ or both demographics have the same base rates:
$$P(Y=1|A=1)=P(Y=1|A=0)$$

# Beyond observational fairness 

- Interpretability
https://geomblog.github.io/fairness/
- Causal ML
