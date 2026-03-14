#include <iostream>
#include <stdio.h>
#include <expected> // For std::expected
#include <string>   // For std::string error type

/**
 * Modernized Discount Calculator
 * This function is a leaf dependency.
 * Uses std::expected for internal error handling and logs errors.
 */
float applyDiscount(float price, float percent) {
    // Define an internal lambda or helper function that can return std::expected.
    // This allows us to encapsulate the error-prone part and use std::expected
    // while maintaining the original function signature.
    auto calculate_discount_internal = [](float p, float pct) -> std::expected<float, std::string> {
        if (p < 0.0f) {
            return std::unexpected("Price cannot be negative.");
        }
        // Assuming a discount percentage should be between 0 and 100 for a "discount".
        // If business rules allow negative percentages (markup) or percentages > 100,
        // these checks would need to be adjusted.
        if (pct < 0.0f || pct > 100.0f) {
            return std::unexpected("Discount percent must be between 0 and 100.");
        }
        return p * (1.0f - (pct / 100.0f));
    };

    auto result = calculate_discount_internal(price, percent);

    if (result.has_value()) {
        return result.value();
    } else {
        // Log the error and return a fallback value to maintain the float return type.
        // Returning the original price means no discount is applied on error,
        // which is a reasonable default behavior for invalid inputs.
        std::cerr << "Error in applyDiscount: " << result.error()
                  << " (Price: " << price << ", Percent: " << percent << "). Returning original price." << std::endl;
        return price;
    }
}

/**
 * Main Processing Logic
 * Uses raw pointers and C-style iteration.
 */
void processOrders() {
    int count = 3;
    float* prices = new float[3]; // Manual allocation
    prices[0] = 100.0;
    prices[1] = 200.0;
    prices[2] = 300.0;

    printf("Processing %d items...\n", count);
    
    float total = 0;
    for (int i = 0; i < count; i++) {
        // Nested dependency call
        float finalPrice = applyDiscount(prices[i], 10.0f);
        total += finalPrice;
    }

    std::cout << "Total after 10% discount: " << total << std::endl;

    delete[] prices; // Manual deallocation
}


int main() {
    processOrders();
    return 0;
}