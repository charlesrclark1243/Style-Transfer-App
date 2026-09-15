export interface StylePreset {
  name: string;
  url: string;
}

// Public-domain paintings from Wikimedia Commons, served from public/presets
export const STYLE_PRESETS: StylePreset[] = [
  { name: 'The Starry Night', url: '/presets/starry-night.jpg' },
  { name: 'The Great Wave off Kanagawa', url: '/presets/great-wave.jpg' },
  { name: 'The Scream', url: '/presets/the-scream.jpg' },
  { name: 'Composition VII', url: '/presets/composition-vii.jpg' },
  { name: 'Water Lilies', url: '/presets/water-lilies.jpg' },
];
