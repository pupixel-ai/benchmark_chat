import DifficultCaseReviewView from "./difficult-case-review-view";


export default function ReflectionDifficultCaseReviewPage({
  params,
}: {
  params: { caseId: string };
}) {
  return <DifficultCaseReviewView caseId={params.caseId} />;
}
