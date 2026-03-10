#include <iostream>
#include <stdio.h>

/**
 * Legacy Discount Calculator
 * This function is a leaf dependency.
 */
float applyDiscount(float price, float percent) {
    return price * (1.0f - (percent / 100.0f));
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

/**
 * ORPHAN FUNCTION
 * This is never called by main or processOrders.
 * The Pruner Node should remove this.
 */
void unusedLogic() {
    std::cout << "This should be pruned!" << std::endl;
}

int main() {
    processOrders();
    return 0;
}