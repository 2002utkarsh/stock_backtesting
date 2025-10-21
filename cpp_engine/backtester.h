#ifndef BACKTESTER_H
#define BACKTESTER_H

#include <cstddef>

// Represents a single bar of price data (e.g., one day of data).
struct StockTick {
    long long timestamp;
    double open;
    double high;
    double low;
    double close;
    int volume;
};

// Manages our trading account's state.
class Portfolio {
public:
    double initial_cash;
    double cash;
    int holdings;

    Portfolio(double initial_cash)
        : initial_cash(initial_cash), cash(initial_cash), holdings(0) {}

    // Updates portfolio value and executes a buy order (simple 1 share per signal).
    void execute_buy(const StockTick& tick) {
        if (cash >= tick.close) {
            holdings += 1;
            cash -= tick.close;
        }
    }

    // Updates portfolio value and executes a sell order (simple 1 share per signal).
    void execute_sell(const StockTick& tick) {
        if (holdings > 0) {
            holdings -= 1;
            cash += tick.close;
        }
    }

    // Calculates the current total value of the portfolio.
    double get_total_value(const StockTick& tick) const {
        return cash + holdings * tick.close;
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Exposed C API for Python ctypes.
// ticks: pointer to first StockTick
// num_ticks: number of ticks
// signals: pointer to int signals (1 buy, -1 sell, 0 hold)
// portfolio_history: pre-allocated double array of length num_ticks (output)
void perform_backtest(const StockTick* ticks, int num_ticks, const int* signals, double* portfolio_history);

#ifdef __cplusplus
}
#endif

#endif // BACKTESTER_H
