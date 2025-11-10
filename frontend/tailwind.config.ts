import type { Config } from 'tailwindcss'

export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', 'sans-serif'],
        mono: ['"Space Mono"', 'monospace'],
      },
      borderRadius: {
        lg: "0px",
        md: "0px",
        sm: "0px",
      },
      boxShadow: {
        'brutal': '4px 4px 0px 0px hsl(0 0% 0% / 1.00)',
        'brutal-sm': '2px 2px 0px 0px hsl(0 0% 0% / 1.00)',
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config
