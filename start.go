package main

import (
	"fmt"
	"log"
	"net/http"

	api "you2api/api" // Please replace with your actual project name
	config "you2api/config"
	proxy "you2api/proxy"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("Run error: %v", err)
	}
}

func run() error {
	// Load configuration
	config, err := config.Load()
	if (err != nil) {
		return fmt.Errorf("Failed to load config: %w", err)
	}

	// If proxy is enabled
	if config.Proxy.EnableProxy {
		proxy, err := proxy.NewProxy(config.Proxy.ProxyURL, config.Proxy.ProxyTimeoutMS)
		if err != nil {
			return fmt.Errorf("Failed to initialize proxy: %w", err)
		}

		// Register proxy handler
		http.Handle("/proxy/", http.StripPrefix("/proxy", proxy))
	}

	// Register API handler to root path
	http.HandleFunc("/", api.Handler)

	port := fmt.Sprintf(":%d", config.Port)
	fmt.Printf("Server is running on http://0.0.0.0%s\n", port)

	// Start server
	if err := http.ListenAndServe("0.0.0.0"+port, nil); err != nil {
		return fmt.Errorf("Failed to start server: %w", err)
	}
	return nil
}
