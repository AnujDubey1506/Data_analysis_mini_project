# =============================================================
# SmartMart Inventory Analysis Project
# =============================================================
# Story:
# You work at "SmartMart" retail store. The manager gave you
# data of 30 products and asked you to analyze it.
# Your job is to find useful insights from this data.
#
# Tools: Pandas, NumPy, Matplotlib, Seaborn
# Dataset: synthetic_data.csv
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -- Loading the dataset --
df = pd.read_csv("synthetic_data.csv")

print("===== SmartMart Inventory Data =====")
print(df.head())
print()
print("Total Products:", len(df))
print("Columns:", list(df.columns))
print()


# ---- TASK 1: How many products are in each category? ----

print("--- Task 1: Products per Category ---")
category_count = df["Category"].value_counts()
print(category_count)

plt.figure()
plt.bar(category_count.index, category_count.values, color="skyblue", edgecolor="black")
plt.title("Task 1: Products per Category")
plt.xlabel("Category")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.savefig("task_01_products_per_category.png", dpi=150)
plt.show()

print("Insight: Each category has 6 products. The store is balanced.")
print()


# ---- TASK 2: What is the average price in each category? ----

print("--- Task 2: Average Price per Category ---")
avg_price = df.groupby("Category")["Price"].mean()
print(avg_price.round(2))

plt.figure()
plt.bar(avg_price.index, avg_price.values, color="salmon", edgecolor="black")
plt.title("Task 2: Average Price per Category")
plt.xlabel("Category")
plt.ylabel("Average Price (Rs)")
plt.tight_layout()
plt.savefig("task_02_avg_price.png", dpi=150)
plt.show()

print("Insight: Electronics is the most expensive category.")
print("Food is the cheapest category.")
print()


# ---- TASK 3: Which are the top 5 expensive products? ----

print("--- Task 3: Top 5 Expensive Products ---")
top5 = df.nlargest(5, "Price")
print(top5[["Product", "Category", "Price"]])

plt.figure()
plt.barh(top5["Product"], top5["Price"], color="orange", edgecolor="black")
plt.title("Task 3: Top 5 Most Expensive Products")
plt.xlabel("Price (Rs)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("task_03_top5_expensive.png", dpi=150)
plt.show()

print("Insight: All top 5 expensive products are from Electronics.")
print()


# ---- TASK 4: Which are the top 5 cheapest products? ----

print("--- Task 4: Top 5 Cheapest Products ---")
bottom5 = df.nsmallest(5, "Price")
print(bottom5[["Product", "Category", "Price"]])

plt.figure()
plt.barh(bottom5["Product"], bottom5["Price"], color="lightgreen", edgecolor="black")
plt.title("Task 4: Top 5 Cheapest Products")
plt.xlabel("Price (Rs)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("task_04_top5_cheapest.png", dpi=150)
plt.show()

print("Insight: All cheapest products are from the Food category.")
print()


# ---- TASK 5: Total stock in each category ----

print("--- Task 5: Total Stock per Category ---")
total_stock = df.groupby("Category")["Stock"].sum()
print(total_stock)

plt.figure()
plt.bar(total_stock.index, total_stock.values, color="plum", edgecolor="black")
plt.title("Task 5: Total Stock per Category")
plt.xlabel("Category")
plt.ylabel("Total Stock (units)")
plt.tight_layout()
plt.savefig("task_05_total_stock.png", dpi=150)
plt.show()

print("Insight: Food has the highest stock. Electronics has the lowest.")
print()


# ---- TASK 6: Average rating per category ----

print("--- Task 6: Average Rating per Category ---")
avg_rating = df.groupby("Category")["Rating"].mean()
print(avg_rating.round(2))

plt.figure()
plt.barh(avg_rating.index, avg_rating.values, color="lightyellow", edgecolor="black")
plt.title("Task 6: Average Rating per Category")
plt.xlabel("Average Rating")
plt.tight_layout()
plt.savefig("task_06_avg_rating.png", dpi=150)
plt.show()

print("Insight: Electronics has the best average rating (4.68).")
print("Food has the lowest average rating (4.07).")
print()


# ---- TASK 7: Products with low stock (less than 100) ----

print("--- Task 7: Low Stock Products (Stock < 100) ---")
low_stock = df[df["Stock"] < 100].sort_values("Stock")
print(low_stock[["Product", "Category", "Stock"]])

plt.figure()
plt.barh(low_stock["Product"], low_stock["Stock"], color="tomato", edgecolor="black")
plt.title("Task 7: Low Stock Products (Need Restocking!)")
plt.xlabel("Stock (units)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("task_07_low_stock.png", dpi=150)
plt.show()

print(f"Insight: {len(low_stock)} products have stock below 100 units.")
print("These products need restocking soon.")
print()


# ---- TASK 8: Products with high rating (4.5 and above) ----

print("--- Task 8: High Rated Products (Rating >= 4.5) ---")
high_rated = df[df["Rating"] >= 4.5].sort_values("Rating", ascending=False)
print(high_rated[["Product", "Category", "Rating"]])

plt.figure()
plt.bar(high_rated["Product"], high_rated["Rating"], color="gold", edgecolor="black")
plt.title("Task 8: High Rated Products (Rating >= 4.5)")
plt.xlabel("Product")
plt.ylabel("Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("task_08_high_rated.png", dpi=150)
plt.show()

print("Insight: Most high-rated products come from Electronics and Sports.")
print()


# ---- TASK 9: Relationship between Price and Rating ----

print("--- Task 9: Price vs Rating ---")
correlation = df["Price"].corr(df["Rating"])
print(f"Correlation between Price and Rating: {round(correlation, 2)}")

plt.figure()
plt.scatter(df["Price"], df["Rating"], color="steelblue", edgecolor="black", s=80)
plt.title("Task 9: Price vs Rating (Scatter Plot)")
plt.xlabel("Price (Rs)")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig("task_09_price_vs_rating.png", dpi=150)
plt.show()

print("Insight: There is a moderate positive correlation (0.56).")
print("Higher priced products tend to have slightly better ratings.")
print()


# ---- TASK 10: Relationship between Price and Stock ----

print("--- Task 10: Price vs Stock ---")
correlation2 = df["Price"].corr(df["Stock"])
print(f"Correlation between Price and Stock: {round(correlation2, 2)}")

plt.figure()
plt.scatter(df["Price"], df["Stock"], color="coral", edgecolor="black", s=80)
plt.title("Task 10: Price vs Stock (Scatter Plot)")
plt.xlabel("Price (Rs)")
plt.ylabel("Stock (units)")
plt.tight_layout()
plt.savefig("task_10_price_vs_stock.png", dpi=150)
plt.show()

print("Insight: Expensive products have less stock.")
print("Cheap products are stocked in higher quantity.")
print()


# ---- TASK 11: Inventory Value (Price x Stock) per category ----

print("--- Task 11: Inventory Value per Category ---")
df["Inventory_Value"] = df["Price"] * df["Stock"]
inv_value = df.groupby("Category")["Inventory_Value"].sum()
print(inv_value.round(2))

plt.figure()
plt.bar(inv_value.index, inv_value.values, color="mediumpurple", edgecolor="black")
plt.title("Task 11: Inventory Value per Category")
plt.xlabel("Category")
plt.ylabel("Inventory Value (Rs)")
plt.tight_layout()
plt.savefig("task_11_inventory_value.png", dpi=150)
plt.show()

print("Insight: Home category has the most money locked in inventory.")
print()


# ---- TASK 12: Overall summary statistics ----

print("--- Task 12: Overall Summary ---")
print()
print("Basic Statistics of Price:")
print(f"  Mean   : Rs {np.mean(df['Price']):.2f}")
print(f"  Median : Rs {np.median(df['Price']):.2f}")
print(f"  Min    : Rs {np.min(df['Price']):.2f}")
print(f"  Max    : Rs {np.max(df['Price']):.2f}")
print(f"  Std Dev: Rs {np.std(df['Price']):.2f}")
print()
print("Basic Statistics of Rating:")
print(f"  Mean   : {np.mean(df['Rating']):.2f}")
print(f"  Median : {np.median(df['Rating']):.2f}")
print(f"  Min    : {np.min(df['Rating']):.2f}")
print(f"  Max    : {np.max(df['Rating']):.2f}")
print()

# Final comparison chart - avg price vs avg rating side by side
categories = df["Category"].unique()
avg_prices = df.groupby("Category")["Price"].mean()
avg_ratings = df.groupby("Category")["Rating"].mean()

x = np.arange(len(categories))  # using numpy for bar positions
width = 0.35

fig, ax1 = plt.subplots()

bars1 = ax1.bar(x - width/2, avg_prices[categories], width, label="Avg Price", color="skyblue", edgecolor="black")
ax1.set_xlabel("Category")
ax1.set_ylabel("Average Price (Rs)", color="steelblue")
ax1.set_xticks(x)
ax1.set_xticklabels(categories)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, avg_ratings[categories], width, label="Avg Rating", color="salmon", edgecolor="black")
ax2.set_ylabel("Average Rating", color="tomato")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("Task 12: Price vs Rating Comparison by Category")
plt.tight_layout()
plt.savefig("task_12_summary.png", dpi=150)
plt.show()

print("Insight: Electronics has both the highest price AND best rating.")
print("Food has the lowest price AND lowest rating.")
print()

# cleanup
df.drop(columns=["Inventory_Value"], inplace=True)


# ===== PROJECT COMPLETE =====
print("=" * 50)
print("  PROJECT SUMMARY")
print("=" * 50)
print("""
 Task  1 - Products per Category       (Bar Chart)
 Task  2 - Average Price per Category   (Bar Chart)
 Task  3 - Top 5 Expensive Products     (Horizontal Bar)
 Task  4 - Top 5 Cheapest Products      (Horizontal Bar)
 Task  5 - Total Stock per Category     (Bar Chart)
 Task  6 - Average Rating per Category  (Horizontal Bar)
 Task  7 - Low Stock Products           (Horizontal Bar)
 Task  8 - High Rated Products          (Bar Chart)
 Task  9 - Price vs Rating              (Scatter Plot)
 Task 10 - Price vs Stock               (Scatter Plot)
 Task 11 - Inventory Value by Category  (Bar Chart)
 Task 12 - Overall Summary              (Grouped Bar)

 Charts saved as task_01 to task_12 PNG files.
 Tools: Pandas, NumPy, Matplotlib, Seaborn
""")
print("=" * 50)
print("  Analysis Complete!")
print("=" * 50)
