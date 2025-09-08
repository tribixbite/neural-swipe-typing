/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./public/**/*.{html,js}",
    "./src/**/*.{ts,tsx,js,jsx}"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'swipe': '#4F46E5',
        'swipe-dark': '#312E81'
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'SF Mono', 'Consolas', 'monospace']
      }
    },
  },
  plugins: [],
}