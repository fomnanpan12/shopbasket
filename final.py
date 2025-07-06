import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def load_data():
    """Load and preprocess the grocery dataset"""
    df = pd.read_csv("Groceries_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by=['Member_number', 'Date'])
    return df

def prepare_transactions(df):
    """Group items by member into transaction lists"""
    transactions = df.groupby('Member_number')['itemDescription'].agg(list).reset_index()
    transactions.rename(columns={'itemDescription': 'Items'}, inplace=True)
    return transactions

def generate_rules(transactions):
    """Generate association rules from transactions"""
    te = TransactionEncoder()
    df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules

def recommend_items(rules, current_items, top_n=3):
    """Generate recommendations based on association rules"""
    matched_rules = rules[
        rules['antecedents'].apply(lambda x: set(current_items).issuperset(x))
    ]
    matched_rules = matched_rules.sort_values('confidence', ascending=False)
    recommendations = set()
    
    for _, rule in matched_rules.head(top_n).iterrows():
        recommendations.update(rule['consequents'])
    
    return list(recommendations)

def main():
    st.title("🛒 Grocery Recommendation System")
    
    # Load and process data
    with st.spinner('Loading data and generating recommendations...'):
        df = load_data()
        transactions = prepare_transactions(df)
        rules = generate_rules(transactions['Items'].tolist())
    
    # Sidebar controls
    st.sidebar.header("Recommendation Settings")
    selected_items = st.sidebar.multiselect(
        "Select items in cart:",
        options=sorted(df['itemDescription'].unique()),
        default=['soda']
    )
    top_n = st.sidebar.slider("Number of recommendations:", 1, 10, 3)
    
    # Main content
    if selected_items:
        recommendations = recommend_items(rules, selected_items, top_n)
        if recommendations:
            st.success("Recommended items to add to your cart:")
            for i, item in enumerate(recommendations, 1):
                st.write(f"{i}. {item}")
        else:
            st.warning("No recommendations found for selected items.")
    else:
        st.info("Please select items from the sidebar to get recommendations.")
    
    # Show raw rules if needed
    if st.checkbox("Show association rules data"):
        st.dataframe(rules.sort_values('confidence', ascending=False))

if __name__ == "__main__":
    main()