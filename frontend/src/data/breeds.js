export const breeds = [
  {
    id: "gir",
    name: "Gir",
    type: "Cattle",
    origin: "Gujarat, India",
    shortDescription:
      "A resilient indigenous dairy breed valued for tropical adaptability, disease tolerance, and stable milk quality in warm climates.",
    characteristics:
      "Long pendulous ears, domed forehead, reddish coat patterns, and a calm, manageable temperament under field conditions.",
    milkProduction: "10-15 liters/day (can vary by feed and management)",
    image: "/images/gir.jpg",
  },
  {
    id: "holstein-friesian",
    name: "Holstein Friesian",
    type: "Cattle",
    origin: "Netherlands",
    shortDescription:
      "A globally dominant commercial dairy breed selected for high milk yield and efficient performance in organized farm systems.",
    characteristics:
      "Distinct black-and-white coat, large body frame, high feed intake capacity, and excellent lactation performance potential.",
    milkProduction: "20-35+ liters/day in intensive systems",
    image: "/images/holstein_friesian.jpg",
  },
  {
    id: "jersey",
    name: "Jersey",
    type: "Cattle",
    origin: "Channel Island of Jersey",
    shortDescription:
      "A compact dairy cattle breed known for rich milk composition, especially high butterfat and solids-not-fat content.",
    characteristics:
      "Light brown to fawn coat, comparatively smaller frame, efficient feed conversion, and docile farm behavior.",
    milkProduction: "12-20 liters/day with high fat percentage",
    image: "/images/jersey.jpg",
  },
  {
    id: "sahiwal",
    name: "Sahiwal",
    type: "Cattle",
    origin: "Punjab region (India/Pakistan)",
    shortDescription:
      "A hardy tropical dairy breed that performs reliably under heat stress and low-input management environments.",
    characteristics:
      "Reddish-dun coat, loose skin, strong parasite tolerance, and robust survival under challenging weather conditions.",
    milkProduction: "8-14 liters/day in tropical conditions",
    image: "/images/sahiwal.jpg",
  },
  {
    id: "murrah",
    name: "Murrah",
    type: "Buffalo",
    origin: "Haryana, India",
    shortDescription:
      "A premium dairy buffalo breed widely preferred for high-fat milk production and excellent market value.",
    characteristics:
      "Jet-black body color, tightly curved horns, deep barrel body, and strong lactation consistency.",
    milkProduction: "8-16 liters/day with high fat content",
    image: "/images/murrah.jpg",
  },
  {
    id: "jaffrabadi",
    name: "Jaffrabadi",
    type: "Buffalo",
    origin: "Gujarat, India",
    shortDescription:
      "A massive buffalo breed suitable for both dairy and draft support, known for sturdiness and field strength.",
    characteristics:
      "Broad forehead, heavy drooping horns, strong skeletal frame, and notable endurance in local farming systems.",
    milkProduction: "6-12 liters/day depending on management",
    image: "/images/jaffrabadi.jpg",
  },
];

export const getBreedById = (id) => breeds.find((breed) => breed.id === id);
