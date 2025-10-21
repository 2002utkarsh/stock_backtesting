#include "backtester.h"
#include <vector>
#include <iostream>

// Core simulation function (internal).
static void run_simulation(
    const StockTick* ticks,
    int num_ticks,
    const int* signals, // 1 for Buy, -1 for Sell, 0 for Hold
    double* portfolio_history // Output array to store results
) {
    Portfolio portfolio(10000.0); // Start with $10,000 cash

    for (int i = 0; i < num_ticks; ++i) {
        const StockTick& current_tick = ticks[i];

        // Execute trade based on the signal for the current day.
        if (signals[i] == 1) { // Buy Signal
            portfolio.execute_buy(current_tick);
        } else if (signals[i] == -1) { // Sell Signal
            portfolio.execute_sell(current_tick);
        }

        // Record the total portfolio value for this time step.
        portfolio_history[i] = portfolio.get_total_value(current_tick);
    }
}

// Exported C-style function callable from Python via ctypes.
// Keep the symbol name stable and C linkage.
extern "C" {
    void perform_backtest(const StockTick* ticks, int num_ticks, const int* signals, double* portfolio_history) {
        run_simulation(ticks, num_ticks, signals, portfolio_history);
    }
}
