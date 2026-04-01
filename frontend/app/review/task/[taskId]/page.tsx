import TaskReviewView from "./task-review-view";


export default function ReflectionTaskReviewPage({
  params,
}: {
  params: { taskId: string };
}) {
  return <TaskReviewView taskId={params.taskId} />;
}
