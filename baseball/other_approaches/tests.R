library(DescTools)

# Read in data
df_iaat <- read.csv('/Users/mymac/inverse_aat/baseball/data/baseball_genetic_rf_errors.csv', header=FALSE)
df_iaat <- cbind(df_iaat, rep('IAAT', nrow(df_iaat)))
df_rf <- read.csv('/Users/mymac/inverse_aat/baseball/data/RF_errors.csv', header=FALSE)
df_rf <- cbind(df_rf, rep('RF', nrow(df_rf)))
df_lr <- read.csv('/Users/mymac/inverse_aat/baseball/data/LR_errors.csv', header=FALSE)
df_lr <- cbind(df_lr, rep('LR', nrow(df_lr)))
df_knn <- read.csv('/Users/mymac/inverse_aat/baseball/data/KNN_errors.csv', header=FALSE)
df_knn <- cbind(df_knn, rep('KNN', nrow(df_knn)))
df_dt <- read.csv('/Users/mymac/inverse_aat/baseball/data/DT_errors.csv', header=FALSE)
df_dt <- cbind(df_dt, rep('DT', nrow(df_dt)))

# Combine errors from different approaches into one dataframe
colnames(df_iaat) <- c('Errors', 'Approach') 
colnames(df_rf) <- c('Errors', 'Approach') 
colnames(df_lr) <- c('Errors', 'Approach') 
colnames(df_knn) <- c('Errors', 'Approach') 
colnames(df_dt) <- c('Errors', 'Approach') 
df <- rbind(df_iaat, df_rf, df_lr, df_knn, df_dt)

# Test for significance
model <- lm(df$Errors ~ df$Approach)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df$Approach", conf.level=0.95)
print(tukey)
