import { ArrowRight, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";

function HomePage() {
  return (
    <div className="space-y-8">
      <section className="card p-8 sm:p-10">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-emerald-700">
          AI-Based Animal Type Classification
        </p>
        <h2 className="mt-2 max-w-3xl text-3xl font-bold leading-tight text-gray-900 sm:text-4xl">
          Image-Based Classification for Cattle and Buffaloes
        </h2>
        <p className="mt-4 max-w-3xl text-base text-gray-600">
          This system helps identify whether an uploaded animal image belongs to cattle or buffalo,
          then stores results for analysis. It combines deep learning inference with a modern web
          dashboard for practical, real-time use.
        </p>
        <p className="mt-3 max-w-3xl text-base text-gray-600">
          In practical dairy operations, quick and correct type identification supports better breed
          tracking, feeding strategy decisions, milk productivity planning, and cleaner farm records.
          This dashboard is designed to make those decisions easier and more data-driven.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            to="/classify"
            className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700"
          >
            Start Classification
            <ArrowRight size={16} />
          </Link>
          <Link
            to="/breeds"
            className="inline-flex items-center rounded-xl border border-gray-300 bg-white px-5 py-3 text-sm font-semibold text-gray-700 transition hover:bg-gray-100"
          >
            Explore Breeds
          </Link>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <article className="card overflow-hidden">
          <img
            src="/images/holstein_friesian.jpg"
            alt="Cattle"
            className="h-56 w-full object-cover"
          />
          <div className="space-y-3 p-6">
            <h3 className="text-2xl font-semibold text-gray-900">Cattle</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Usually slimmer body shape with varied horn forms.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Includes major dairy breeds like Holstein Friesian and Jersey.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Milk often has lower fat than buffalo milk.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Common dairy cattle breeds are optimized for volume and can adapt to different management systems.
              </li>
            </ul>
          </div>
        </article>

        <article className="card overflow-hidden">
          <img
            src="/images/murrah.jpg"
            alt="Buffalo"
            className="h-56 w-full object-cover"
          />
          <div className="space-y-3 p-6">
            <h3 className="text-2xl font-semibold text-gray-900">Buffalo</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Typically darker coat, bulkier frame, and heavy horns.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Key breeds include Murrah and Jaffrabadi.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Milk is generally richer with higher fat percentage.
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 size={16} className="mt-0.5 text-emerald-600" />
                Buffalo breeds are often preferred where rich milk quality and fat-based dairy products are prioritized.
              </li>
            </ul>
          </div>
        </article>
      </section>
    </div>
  );
}

export default HomePage;
