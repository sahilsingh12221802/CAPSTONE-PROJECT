import BreedCard from "../components/BreedCard";
import { breeds } from "../data/breeds";

function BreedsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Breeds Explorer</h2>
        <p className="mt-2 text-gray-600">
          Browse common cattle and buffalo breeds used in this classification project.
        </p>
        <p className="mt-2 max-w-4xl text-sm text-gray-600">
          Each profile below summarizes appearance patterns, region of origin, and dairy relevance.
          These summaries are intentionally practical so you can quickly compare type-specific traits
          before running image classification.
        </p>
      </div>

      <section className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {breeds.map((breed) => (
          <BreedCard key={breed.id} breed={breed} />
        ))}
      </section>
    </div>
  );
}

export default BreedsPage;
