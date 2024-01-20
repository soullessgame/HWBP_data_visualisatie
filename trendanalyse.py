import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#read data function
def read_data(file_name):
    """
    Reads the excel file and returns a pandas dataframe.
    """
    try:
        df = pd.read_excel(file_name, engine='openpyxl', sheet_name=0)
        return df
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None


#preprcess data, dataframes etc.
def preprocess_data(df):
    """
    Preprocesses the data by renaming columns, dropping unnecessary columns, and filtering data.
    """
    col_names = ['project', 'programmajaar', 'ranking', 'projectcode', 'beheerder', 'projectnaam',
                 'lengte', 'kunstwerken', 'normvak', 'samengevoegd', 'afgesplitst',
                 'begin_VV', 'eind_VV', 'begin_PU', 'eind_PU', 'begin_R', 'Eind_R',
                 'duur_VV', 'duur_PU', 'duur_R', 'duur_totaal',
                 'netto_vertraging', 'vertraging2018_2021', 'vertraging2018_2022',
                 'kosten_VV', 'kosten_PU', 'kosten_R', 'kosten_totaal', 'deltakosten',
                 'deltakosten2018_2021', 'deltakosten2018_2022', 'deltalengte2018_2021',
                 'deltalengte2018_2022', 'efficiency2018_2021', 'efficiency2018_2022', 'grootin2021',
                 'grootin2022', 'VV2020', 'PU2020', 'R2020', 'jaren_tot_R', 'efficiency', 'tijd_tot_VKA', 'Voorbij_VKA',
                 'voorbij_VKA_naam', 'Groot_in_2022_naam']
    df.columns = col_names
    to_drop = ['kunstwerken', 'normvak', 'projectcode', 'samengevoegd', 'afgesplitst']
    df.drop(to_drop, inplace=True, axis=1)
    df.drop(df[df['programmajaar'] < 18].index, inplace=True)
    df.reset_index(inplace=True)
    df["lengte_groep"] = np.where(df["lengte"] >= 9000, "kleine scope", "grote scope")
    return df


def process_2018_data(df):
    """
    Processes data for the year 2018.
    """
    df_2018 = df[df['programmajaar'] == 18]
    # Filter for projects in 2018, Calculate the median of kosten_totaal for 2018
    median_kosten_2018 = df_2018['kosten_totaal'].median()
    # Filter for large projects: those with kosten_totaal above the median
    grote_projecten18 = df_2018[df_2018['kosten_totaal'] > median_kosten_2018]
    kleine_projecten18 = df_2018[df_2018['kosten_totaal'] <= median_kosten_2018]
    return grote_projecten18, kleine_projecten18, df_2018


def process_2022_data(df):
    """
    Processes data for the year 2022.
    """
    # Filter for projects in 2018, Calculate the median of kosten_totaal for 2018
    df_2022 = df[df['programmajaar'] == 22]
    median_kosten_2022 = df_2022['kosten_totaal'].median()
    # Filter for large projects: those with kosten_totaal above the median
    grote_projecten22 = df_2022[df_2022['kosten_totaal'] > median_kosten_2022]
    # Filter for (small projects: those with kosten_totaal less than or equal to the median
    kleine_projecten22 = df_2022[df_2022['kosten_totaal'] <= median_kosten_2022]
    return grote_projecten22, kleine_projecten22, df_2022


#preparing data
df = read_data('Python_trend_analyse_extra_test2.xlsx')
df = preprocess_data(df)
grote_projecten18, kleine_projecten18, df_2018 = process_2018_data(df)
grote_projecten22, kleine_projecten22, df_2022 = process_2022_data(df)

#set plot style
sns.set_palette("deep")
sns.set_style('darkgrid')

# Descriptieve statistieken van de projectduratie en vertragingen
print(df.groupby('programmajaar')['duur_totaal'].describe())
print(df.groupby('programmajaar')['netto_vertraging'].mean())
print(df.groupby('programmajaar')['netto_vertraging'].mean().sum())
print(df.groupby('programmajaar')['duur_VV'].describe())
print(df.groupby('programmajaar')['duur_VV'].describe().mean())
print(df.groupby('programmajaar')['duur_PU'].describe())
print(df.groupby('programmajaar')['duur_R'].describe())
VV_means = np.array(df.groupby('programmajaar')['kosten_VV'].mean())
PU_means = np.array(df.groupby('programmajaar')['kosten_PU'].mean())
R_means = np.array(df.groupby('programmajaar')['kosten_R'].mean())
print(df.groupby('programmajaar')['kosten_totaal'].describe())
table = pd.pivot_table(df, values='duur_totaal', index=['programmajaar'], columns=["grootin2022"], aggfunc=np.std)
print(table)


# Visualization of total project duration per year
ax = sns.violinplot(x="programmajaar", y="duur_totaal", data=df, palette="muted")
ax.set(xlabel='Programmajaar', ylabel='Duur projecten')
ax.set_title('Spreiding projectduur per programmajaar')
plt.show()

ax = sns.catplot(x="duur_totaal", kind="count",col='programmajaar', palette="ch:.25", data=df)
ax.set(xlabel='Programmajaar', ylabel='Duur projecten')
plt.show()
plt.savefig('duur_programma_jaar.png', dpi=300)

# Visualization related to project size in 2022
ax = sns.boxplot(x="programmajaar", y='duur_totaal', hue='grootin2022', data=df, palette="Set3")
ax.set(xlabel='Programmajaar', ylabel='Duur projecten')
ax.set_title('Totale uitloop van projeten naar projectkosten')
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["Klein project in 2022", "groot project in 2022"])
plt.show()

ax = sns.catplot(x="programmajaar", y="duur_totaal", hue='grootin2022' , kind="point", data=df, markers=["o", "x"], linestyles = ["-", "--"], height=5, aspect=11.7/8.27)
ax.set(xlabel='Programmajaar - N=40', ylabel='duur van projecten in jaren')
new_title ='projectgrootte'
ax._legend.set_title(new_title)
new_labels = ['Klein project', 'Groot project']
for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
ax.fig.subplots_adjust(right=0.78, left=0.0)
plt.savefig('kleingrootduur.png')
plt.show()

# Visualization related to project size in 2018
ax = sns.kdeplot(grote_projecten18['duur_totaal'], shade=True, label='groot 18')
ax = sns.kdeplot(kleine_projecten18['duur_totaal'], shade=True, label='klein 18')
plt.xlim(0, 17.5)
plt.ylim(0, 0.4)
plt.axvline(x=grote_projecten18['duur_totaal'].median(), color='cornflowerblue', linestyle='--', label='Mediaan groot')
plt.axvline(x=kleine_projecten18['duur_totaal'].median(), color='orange', linestyle='--', label='Mediaan klein')
ax.set(xlabel='Duur van projecten in jaren', ylabel='Kansverdelingsfunctie')
ax.set_title('Projectduur van grote en kleine projecten - 18')
ax.legend()
plt.show()

ax = sns.kdeplot(grote_projecten22['duur_totaal'], shade=True, label='groot 22')
ax = sns.kdeplot(kleine_projecten22['duur_totaal'], shade=True, label='klein 22')
plt.xlim(0, 17.5)
plt.ylim(0, 0.4)
plt.axvline(x=grote_projecten22['duur_totaal'].median(), color='cornflowerblue', linestyle='--', label='Mediaan groot')
plt.axvline(x=kleine_projecten22['duur_totaal'].median(), color='orange', linestyle='--', label='Mediaan klein')
ax.set(xlabel='Duur van projecten in jaren', ylabel='Kansverdelingsfunctie')
ax.set_title('Projectduur van grote en kleine projecten - 22')
ax.legend()
plt.show()

sns.boxplot(x="programmajaar", y='duur_VV', hue='grootin2022', data=df, palette="Set3")
plt.show()

# Create new categories for costs in 2022
median_kosten = df['kosten_totaal'].median()
df['kosten_groep'] = df['kosten_totaal'].apply(lambda x: 'high' if x > median_kosten else 'low')

#Duratie van project uitwerkingsfase naar kosten categoriën 'hoog' en 'laag'
sns.violinplot(x="programmajaar", y="duur_PU",hue='kosten_groep', data=df, palette="Set3", split=True, inner="quart")
plt.show()

#Duratie van realisatiefase naar kosten categoriën 'hoog' en 'laag'
sns.violinplot(x="programmajaar", y="duur_R", hue='kosten_groep', data=df, palette="Set3", density_norm='count')
plt.show()

#Gemiddelde duur van elke projectfase, per programmajaar #bar chart
subset1=df[['programmajaar', 'duur_VV', 'duur_PU', 'duur_R']]
dfnew1=pd.melt(subset1, id_vars="programmajaar", var_name="Projectfase", value_name="Duur van fase in jaren")
sns.catplot(x='programmajaar', y='Duur van fase in jaren', hue='Projectfase', data=dfnew1, kind='bar', errorbar=('ci', 70), palette="Blues_d")
plt.show()

#Gemiddelde duur van elke projectfase, per programmajaar # stacked bar chart
sns.set_palette("Blues_d")
labels=['2018', '2019', '2020', '2021', '2022']
fig, ax = plt.subplots()
ax.bar(labels, VV_means, label='Verkenningsfase')
ax.bar(labels, PU_means, label='Planuitwerkingfase', bottom=VV_means)
ax.bar(labels, R_means, label='Realisatiefase', bottom=VV_means + PU_means)
ax.set_ylabel('Gemiddelde projectduur in jaren')
ax.set_title('projectduur per programmeerjaar')
ax.legend()
plt.savefig('totaleprojectduurstacked.png')
plt.show()

#Gemiddelde projectkosten per programmajaar violin plot.
sns.set_palette("deep")
df.groupby('programmajaar')['kosten_totaal'].describe().to_csv(r'C:\Users\twand\PycharmProjects\practise\tabellen.csv')

ax = sns.violinplot(x="programmajaar", y="kosten_totaal", data=df, inner=None, color=".8")
ax = sns.stripplot(x="programmajaar", y="kosten_totaal", data=df)
ax.set_ylabel('Gemiddelde projectkosten in mln')
ax.set_title('Projectkosten per programmeerjaar')
plt.ylim(-50, 450)
plt.show()


# Histogram van totale kosten 2018 projecten
plt.figure(figsize=(10, 6))
sns.histplot(df_2018['kosten_totaal'], kde=False, bins=20, color='skyblue')
plt.title('Verdeling van Totale Kosten in Programmaplan 2018')
plt.xlabel('Totale Kosten [Mln EUR]')
plt.ylabel('Aantal Projecten')
plt.show()

# Violin plot van projectkosten met jaren, verdeeld over categorieën 'lange projecten' en 'korte projecten'
plt.figure(figsize=(10, 6))
sns.violinplot(x="programmajaar", y="kosten_totaal", hue='lengte_groep', data=df, palette="coolwarm")
plt.title('Projectkosten per Programmajaar en Lengtegroep')
plt.xlabel('Programmajaar')
plt.ylabel('Totale Kosten [Mln EUR]')
plt.show()

# Point plot van totale kosten naar jaar en projectomvang in 2022
plt.figure(figsize=(12, 7))
point_plot = sns.catplot(x="programmajaar", y="kosten_totaal", hue="grootin2022", kind="point",
                         data=df, markers=["o", "x"], linestyles=["-", "--"],
                         palette=["orange", 'cornflowerblue'], height=6, aspect=1.5)
point_plot.set_axis_labels('Programmajaar', 'Totale Kosten [Mln EUR]')
point_plot.fig.suptitle('Totale Kosten per Programmajaar en Projectgrootte in 2022')
point_plot.set(ylim=(0, 270))
plt.show()

# Scatter plot (lmplot) lengte versus totale kosten voor gegevens uit 2018 en 2022
for year_data, year in zip([df_2018, df_2022], ['2018', '2022']):
    lm_plot = sns.lmplot(x="lengte", y="kosten_totaal", hue='grootin2022', lowess=True, data=year_data)
    lm_plot.set_axis_labels('Lengte', 'Totale Kosten [Mln EUR]')
    lm_plot.fig.suptitle(f'Lengte vs. Totale Kosten - {year}')
    plt.show()

# Bar plot van de gemiddelde kosten van elke projectfase per jaar
dfnew = pd.melt(df[['programmajaar', 'kosten_VV', 'kosten_PU', 'kosten_R']],
                id_vars="programmajaar", var_name="Projectfase",
                value_name="Gemiddelde Kosten van Fase in Meuro")
plt.figure(figsize=(12, 7))
sns.barplot(x='programmajaar', y='Gemiddelde Kosten van Fase in Meuro', hue='Projectfase',
            data=dfnew, palette="Blues_d")
plt.title('Gemiddelde Kosten per Projectfase en Programmajaar')
plt.xlabel('Programmajaar')
plt.ylabel('Gemiddelde Kosten [Mln EUR]')
plt.show()

# Gestapeld staafdiagram voor gemiddelde projectduur per fase per jaar
fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(labels, VV_means, label='Verkenningsfase', color='skyblue')
ax.bar(labels, PU_means, bottom=VV_means, label='Planuitwerkingfase', color='steelblue')
ax.bar(labels, R_means, bottom=[i+j for i,j in zip(VV_means, PU_means)], label='Realisatiefase', color='navy')
ax.set_ylabel('Gemiddelde Projectduur in Jaren')
ax.set_xlabel('Programmajaar')
ax.set_title('Gemiddelde Projectduur per Programmeerjaar per Fase')
ax.legend()
plt.show()
