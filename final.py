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

def get_items_with_recommendations(rules):
    """Return only items that have at least one recommendation"""
    items_with_rules = set()
    for _, rule in rules.iterrows():
        items_with_rules.update(rule['antecedents'])
    return sorted(items_with_rules)

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
    st.title("🛒 Smart Grocery Recommender")
    
    # Load and process data
    with st.spinner('Analyzing purchase patterns...'):
        df = load_data()
        transactions = prepare_transactions(df)
        rules = generate_rules(transactions['Items'].tolist())
        recommended_items = get_items_with_recommendations(rules)
    
    # Sidebar controls
    st.sidebar.header("Recommendation Settings")
    selected_items = st.sidebar.multiselect(
        "Select items in your cart:",
        options=recommended_items,  # Only show items that have recommendations
        default=['whole milk'] if 'whole milk' in recommended_items else None
    )
    top_n = st.sidebar.slider("Number of recommendations:", 1, 10, 3)
    
    # Main content
    if selected_items:
        recommendations = recommend_items(rules, selected_items, top_n)
        if recommendations:
            st.success("Recommended additions to your cart:")
            for i, item in enumerate(recommendations, 1):
                st.write(f"{i}. {item}")
        else:
            st.warning("No recommendations found for these items.")
    else:
        st.info("Select items from your cart to get recommendations")
        st.write(f"{len(recommended_items)} items available with known purchase patterns")

    # Show raw rules if needed
    if st.checkbox("Show association rules data"):
        st.dataframe(rules.sort_values('confidence', ascending=False))

if __name__ == "__main__":
    main()