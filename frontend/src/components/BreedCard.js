import { Link } from "react-router-dom";

function BreedCard({ breed }) {
  return (
    <Link
      to={`/breeds/${breed.id}`}
      className="card group overflow-hidden transition hover:-translate-y-1"
    >
      <div className="aspect-[16/11] overflow-hidden">
        <img
          src={breed.image}
          alt={breed.name}
          className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
        />
      </div>
      <div className="space-y-2 p-5">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">{breed.name}</h3>
          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">
            {breed.type}
          </span>
        </div>
        <p className="text-sm text-gray-600">{breed.shortDescription}</p>
      </div>
    </Link>
  );
}

export default BreedCard;
