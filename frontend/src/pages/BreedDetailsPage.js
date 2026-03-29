import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { getBreedById } from "../data/breeds";

function BreedDetailsPage() {
  const { id } = useParams();

  const breed = useMemo(() => getBreedById(id), [id]);

  if (!breed) {
    return (
      <div className="card p-8 text-center">
        <h2 className="text-2xl font-bold text-gray-900">Breed Not Found</h2>
        <p className="mt-2 text-gray-600">The requested breed does not exist in the explorer.</p>
        <Link to="/breeds" className="mt-4 inline-flex rounded-xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white">
          Back to Breeds
        </Link>
      </div>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[2fr,3fr]">
      <div className="card overflow-hidden">
        <img src={breed.image} alt={breed.name} className="h-full min-h-[320px] w-full object-cover" />
      </div>

      <article className="card p-7">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">{breed.type}</p>
        <h2 className="mt-1 text-3xl font-bold text-gray-900">{breed.name}</h2>

        <dl className="mt-6 space-y-4 text-sm">
          <div>
            <dt className="font-semibold text-gray-900">Origin</dt>
            <dd className="mt-1 text-gray-700">{breed.origin}</dd>
          </div>
          <div>
            <dt className="font-semibold text-gray-900">Characteristics</dt>
            <dd className="mt-1 text-gray-700">{breed.characteristics}</dd>
          </div>
          <div>
            <dt className="font-semibold text-gray-900">Milk Production</dt>
            <dd className="mt-1 text-gray-700">{breed.milkProduction}</dd>
          </div>
        </dl>

        <div className="mt-6 rounded-xl border border-gray-200 bg-gray-50 p-4">
          <h3 className="text-base font-semibold text-gray-900">Why This Breed Matters</h3>
          <p className="mt-2 text-sm leading-6 text-gray-700">
            {breed.name} is important in classification pipelines because visual traits, posture,
            horn structure, and body frame strongly influence model predictions. Better breed-level
            understanding helps validate whether classification outputs are biologically reasonable
            and farm-usable.
          </p>
          <p className="mt-2 text-sm leading-6 text-gray-700">
            In real deployments, combining model confidence with breed context improves trust in
            predictions and helps avoid decision errors during farm-level data collection.
          </p>
        </div>
      </article>
    </div>
  );
}

export default BreedDetailsPage;
